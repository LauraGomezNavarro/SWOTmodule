# %filter_SWOT_data.py

# Modules:
import numpy as np
import glob
from netCDF4 import Dataset
import xarray as xr

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

from shutil import copyfile
import os
from pylab import *

from scipy import ndimage as nd

###########################################################

sys.path.insert(0, "/Users/laura/PhD_private/develop/SWOT/")
import SWOT_data_class_upd

###########################################################


# Necessary functions:

def gradi(I): 
    """
    Calculates the gradient in the x-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last row is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])

    M[0:-1,:] = np.subtract(I[1::,:], I[0:-1,:])
    return M

def gradj(I): 
    """
    Calculates the gradient in the y-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last column is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])
    M[:,0:-1] =  np.subtract(I[:,1::], I[:,0:-1])
    return M

def div(px, py): 
    """
    Calculates the divergence of a vector (px, py) and gives M as ouput
    ; where px and py have the some size and are the gradient in x and y respectively 
    of a variable p.
    The x component of M (Mx) first row is = to the first row of px.
    The x component of M (Mx) last row is = to - the before last row of px. (last one = 0)
    The y component of M (My) first column is = to the first column of py.
    The y component of M (My) last column is = to - the before last column of py. (last one = 0)
    ??#(de sorte que div=-(grad)^*)
    """
    m, n = px.shape
    M = np.zeros([m,n])
    Mx = np.zeros([m,n])
    My = np.zeros([m,n])
 
    Mx[1:m-1, :] = px[1:m-1, :] - px[0:m-2, :]
    Mx[0, :] = px[0, :]
    Mx[m-1, :] = -px[m-2, :]

    My[:, 1:n-1] = py[:, 1:n-1] - py[:, 0:n-2]
    My[:, 0] = py[:,0]
    My[:, n-1] = -py[:, n-2]
     
    M = Mx + My;
    return M

def np_laplacian(u):
    """
    Calculates the laplacian of u using the divergence and gradient functions and gives 
    as output Ml.
    """
    Ml = div(gradi(u), gradj(u));
    return Ml

def trilaplacian(u):
    u2 = np_laplacian(np_laplacian(np_laplacian(u)))
    
    return u2

def gaussian_with_nans(u, sigma):
    '''
    Evan Mason
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    '''
    assert np.ma.any(u.mask), 'u must be a masked array'
    mask = np.flatnonzero(u.mask)
    #u.flat[mask] = np.nan

    v = u.data.copy()
    #v.flat[mask] = 0
    v[:] = nd.gaussian_filter(v ,sigma=sigma)

    w = np.ones_like(u.data)
    w.flat[mask] = 0
    w[:] = nd.gaussian_filter(w, sigma=sigma)

    return v/w

def penalization_filter(var, mask_of_data, var_obs, lambd=40, regularization_model=1, iter_max=500, epsilon=1.e-6):
    '''
    SSH_obs_gap
    -Input(s):
    --var = variable to be filtered
    Default = 
    --mask_of_data = array of True and False values.  0 when land or swath gap values.
    Default = 
    --var_obs = orginal data without filling the masked values 
    --lambd = lambda: regularization parameter (the larger lambda, the more regular is the denoised image).  
    This parameter needs to be adapted to the regularization model. 
    --regularization_model = 0 if Tikhonov regularization (first order penalization, i.e. gradient penalization) 
    and 1 if third order penalization (regularization of vorticity)
    --iter_max = maximum number of iterations in case it takes very long to converge.  500 by default.
    --epsilon = 0.000001 (1.e-6 by default)
    -Output(s):
    --ima1 = filtered image
    --itern = number of iterations realized until convergence
    '''
    
    ima0 = var
    ima_obs = var_obs
    
    mask01 = np.ones_like(var)
    mask01[mask_of_data] = 0
    
    if regularization_model == 0:
        tau = 1./(8*lambd)
        operator = np_laplacian
    elif regularization_model == 1:
        tau=1./(512*lambd)
        operator = trilaplacian
    else:
        print "Undefined regularization model"
    
    itern = 1
    ima1 = ima0 + tau*(mask01 * (ima_obs - ima0) + lambd*operator(ima0))

    norm_minus = np.nansum((ima1 - ima0)**2) + np.nansum((ima1 - ima0)**2)
    norm_ima0 = np.nansum((ima0)**2) + np.nansum((ima0)**2)
    conv_crit = norm_minus / norm_ima0

    while (conv_crit > epsilon) & (itern < iter_max):
        ima0 = np.copy(ima1)
        itern = itern + 1
        ima1 = ima0 + tau*(mask01 * (ima_obs - ima0) + lambd*operator(ima0))
        norm_minus = np.nansum((ima1 - ima0)**2) + np.nansum((ima1 - ima0)**2)
        norm_ima0 = np.nansum((ima0)**2) + np.nansum((ima0)**2)
        conv_crit = norm_minus / norm_ima0
            
    return ima1, itern

################################################################
    
def SWOTdenoise(datadir=None, filename=None **kwargs):
    # , method, parameter, inpainting='no',
    """
    Driver function. It is organized as follows:
    1. Read function arguments
        1.1 Read the SWOT data.
    2. Prepare the SWOT data for filtering.
    3. Filter the data.
    4. Save the data??
    FOR THE MOMENT ADAPTED FOR THE BOX DATASET
    Inputs:
    - Input data: 2 options are possible:
        -- swotfile=datadirectory + filename 
        -- Giving the input variables:
            --- lon
            --- lat
            --- data: ssh_initial.  It should be a masked array, even if no data masked.
            --- x_ac: the distance across track
    - Method:  Filtering method to be applied to the SWOT data.  3 methods are available:
        -- Boxcar (generic) filter.  method='boxcar'
        -- Gaussian filter. method='gaussian'
        -- Penalization filter: 
            --- First order penalization. method='penalization_order_1'
            --- Third order penalization. method='penalization_order_3'
    - Parameter: Depends on method used:
        -- boxcar: footprint
        -- gaussian: sigma
        -- penalization filter: 
            --- lamnbda
            --- iter_max
            --- epsilon
    - Inpainting: Decide whether you want the gap between the 2 swaths to be filled (inpainting='yes') or not (inpainting='no')
    Outputs:
    - filtered_data: 
    - itern: the maximum number of iteration used in the penalization filters 
    """
    
    ################################################################
    # 1. Read function arguments:
    # 1.1. Read the data:
    if swotfile == None:
        if any( (ssh_initial, lon, lat, x_ac) ) is False:
            print "You must provide a SWOT file name or SSH, lon, and lat arrays"
            sys.exit()
        else:
            ssh_initial = kwargs.get('data', None)
            lon         = kwargs.get('lon', None)
            lat         = kwargs.get('lat', None)
            x_ac        = kwargs.get('x_ac', None)
    
    else:
        
        filename = swotfile.split('/')[-1]
        datadir = swotfile.split(filename)[0]
        
        swot_pass_b = SWOT_data_class_upd.swot_pass(datadir, filename)
        swotfiles = swot_pass_b.get_list_of_swotfiles()
        # Assuming one file read at a time and for the box dataset and that SSH_obs is the one to be filtered by default not SSH_model:
        print('Input file:')
        print swotfiles[0]
        swot_pass_b.read_data_ncfile_box(swotfiles[0])
        ssh_initial = swot_pass_b.SSH_obs
        lon         = swot_pass_b.lon
        lat         = swot_pass_b.lat
        x_ac        = swot_pass_b.x_ac
           
    method = kwargs.get('method', 'penalization_order_3')
    lambd = kwargs.get('lambda', 1.5)          # default value to be defined 
    itermax = kwargs.get('itermax', 500)
    epsilon = kwargs.get('epsilon', 1.e-6)
    
    inpainting = kwargs.get('inpainting', None)      # Only for quadratic penalization methods
    
    mask_data = ssh_initial.mask
    ################################################################
    # 2. Preparing the SWOT data for filtering:

    ##=================Preparing the SWOT data arrays===============

    # 2.1. Obtain mask of data (ssh_initial):
    mask_data = ssh_initial.mask
    
    if all(mask_data == False):
        'Print mask problem: all False'
        
    # 2.2. Making SWOT array organized as (lon, lat) a.k.a., (x, y) a.k.a., (x_ac, x_al): 
    
    # Concept: x direction (x_ac) will always be smaller (in theory) than in y (x_al), so:
    
    indsx, indsy = lon.shape # lon chosen randomly, any variable ok
    if indsx > indsy:
        lon = np.transpose(lon)
        lat = np.transpose(lat)
        ssh_initial = np.transpose(ssh_initial)
        mask_data = np.transpose(mask_data)

    elif indsx == indsy:
        print 'Problem'
    
    # 2.3. Check longitude in -180 : +180 format:
    if any(xlo < 0 for xlo in lon):
        lon[lon > 180] -= 360 

    # 2.4. Flipping the SWOT array if necessary: 
    # So that lon increases from left to right and lat from bottom to top 
    # (depends on if pass is ascending or descending)

    lon_dif = lon[1,0] - lon[0,0]
    lat_dif = lat[0,1] - lat[0,0]

    # Need to revise below , above already adapted for datasets oragnized as x,y instead of y,x
    if (lat_dif<0):
        print 'flipped variables created'
        ## Ascending pass (pass 22 for in the fast-sampling phase)        
        lon = np.flipud(lon)
        lat = np.flipud(lat)
        ssh_initial = np.flipud(ssh_initial)
        mask_data = np.flipud(mask_data)
        
    elif (lon_dif<0):
        print 'flipped variables created'
        ## Descending pass (pass 9 for in the fast-sampling phase)
        lon = np.fliplr(lon)
        lat = np.fliplr(lat)
        ssh_initial = np.fliplr(ssh_initial)
        mask_data = np.fliplr(mask_data)
        
    # 2.5. Fixing the SWOT grid (regular longitude increments, i.e., no repeated nadir values)
    # At a 1km resolution, should have a size of 121 across track (size of first dim after transposing)    

    xx, yy = lon.shape
    if (lon[(xx/2)-1, 0] == lon [xx/2, 0]): # this only happens
        # when simulation is done with no gap
        # if no gap simulaton case need to eliminate repeated nadir value
        lon = np.delete(lon, (xx/2), axis=0)
        lat = np.delete(lat, (xx/2), axis=0)
        ssh_initial = np.delete(ssh_initial, (xx/2), axis=0)
        x_ac = np.delete(x_ac, (xx/2), axis=0)
        mask_data = np.delete(mask_data, (xx/2), axis=0)        
        
    # 2.6. Eliminating values at x_ac = 0.  (By now (in the code) the x_ac dimension (x) should be the first axis) 
    if any(x_ac == 0):
        x0 = np.where(x_ac==0)

        lon = np.delete(lon, x0, axis=0)
        lat = np.delete(lat, x0, axis=0)
        SSH_model = np.delete(SSH_model, x0, axis=0)
        SSH_obs = np.delete(SSH_obs, x0, axis=0)
        x_ac = np.delete(x_ac, x0, axis=0)
        mask_data = np.delete(mask_data, x0, axis=0)
        
    # 2.7. Masking the SWOT outputs
    ## 2.7.1. Change masked values to 0:

    ssh_initial[mask_data] = 0. 

    ## 2.7.2. Converting variable to masked array
    ssh_initial_ma = np.ma.masked_array(ssh_initial, mask_data)

    if (method == 'penalization_order_1') | (method == 'penalization_order_3'):
    
        ## 2.8. Applying the gaussian filter:

        sigma = 10 # --> this sigma?
        ssh_initial_gausf = gaussian_with_nans(ssh_initial_ma, sigma)

        ## 2.9. Assigning the gaussian filtered data to the gap:

        ssh_initial_gap_full = ssh_initial_ma.copy()

        ssh_initial_gap_full[mask_gap] = ssh_initial_gausf[mask_gap] # or leave it as ssh_initial_nogap.mask? and one variable less?
    
    # At this point we have obtained these variables:
    # -ssh_initial: variable correctly arranged (xx, yy, with dims increasing in expected sense) and gap values = 0 (without them being masked)
    # -ssh_initial_ma: gap masked, with mask values = 0
    # For penalization filter methods, also:
    # -ssh_initial_gausf: gaussian filtered variable
    # -ssh_initial_gap_full: gap filled with gaussian filtered data

    
    ################################################################
    # 3. Applying the filter 

    if method == 'generic':
        ssh_filtered = nd.generic_filter(data, function=np.nanmean, footprint=parameter) # np.nanmean ok? # Remember footprint must be a matrix

    elif method == 'gaussian':
        ssh_filtered = gaussian_with_nans(data, parameter)

    elif method == 'penalization_order_1':
        ssh_filtered, iter_number = penalization_filter(var=data, mask_of_data=mask_gap, var_obs=SSH_obs_ma, lambd=parameter, regularization_model=1)

    elif method == 'penalization_order_3':
        ssh_filtered, iter_number = penalization_filter(var=data, mask_of_data=mask_gap, var_obs=SSH_obs_ma, lambd=parameter, regularization_model=3)
    
    else:
        print('Method not specified')
    
    if inpainting = 'yes':
        ssh_filtered.mask = False
    
    ###############################################################
    # 4. Manage results
    if swotfile is not None:
        # saving outputs as nc file
        "copy initial file and replace SSH with filtered SSH. To be done."
    else:
        # just returning the output variables
        if (method == 'generic') or (method == 'gaussian'):
            return ssh_filtered

        elif (method == 'penalization_order_1') or (method == 'penalization_order_3'):
            return ssh_filtered, itern
        else:
             print('Method not specified')

