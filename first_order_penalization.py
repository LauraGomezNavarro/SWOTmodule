#!/usr/bin/env  python (to edit)
#=======================================================================
""" first_order_penalization.py
Filters SWOT data using a first_order_penalization (Tikhonov regularization) filter for a given lambda and number of iterations
, and saves the filtered data in a netcdf file.
"""
#=======================================================================
import glob
from netCDF4 import Dataset
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

from shutil import copyfile
import os

from scipy import ndimage as nd

#=======================================================================
def gradx(I): 
    """
    Calculates the gradient in the x-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last row is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])

    M[0:-1,:] = np.subtract(I[1::,:], I[0:-1,:])
    return M

def grady(I): 
    """
    Calculates the gradient in the y-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last column is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])
    M[:,0:-1] =  np.subtract(I[:,1::], I[:,0:-1])
    return M

#=======================================================================
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

def laplacian(u):
    """
    Calculates the laplacian of u using the divergence and gradient functions and gives 
    as output Ml.
    """
    Ml = div(gradx(u), grady(u));
    return Ml
#=======================================================================

def filter(u, iter_max, lambd, regularization_model):
    '''
    u = variable to be filtered
    iter_max = maximum number of iterations
    lambd = lambda: regularization parameter (the larger lambda, the more regular is the denoised image).  This parameter needs to be adapted to the regularization model. 
    regularization_model = 0 if Tikhonov regularization and 1 if second order penalization (regularization of vorticity)
    '''
    
    ima = u
    ima_obs = u

    if regularization_model == 0:
        tau = 1./(8*lambd)
        for k in xrange(0,iter_max):
            ima = ima + tau*(ima_obs - ima + lambd*laplacian(ima))
            imaw = imaw + tau*(ima_obsw - imaw + lambd*laplacian(imaw))

    elif regularization_model == 1:
        tau=1./(512*lambd)
        for k in xrange(0,iter_max):
            ima = ima + tau*(ima_obs-ima+lambd*laplacian(laplacian(laplacian(ima))))

    return ima

#=======================================================================

def plot_map(xx, yy, var, ax, box, vmin, vmax, merv, parv, cmap):
    """
    Plots an individual plot of the data using scatter and Basemap.
    Input variables:
    - xx = x-axis variable, in this case, longitude
    - yy = y-axis variable, in this case, latitude
    - var = variable to be plotted
    - ax = axis
    - box = box region of the area wanted to be shown
    , where box is a 1 x 4 array: 
    [minimum_longitude maximum_longitude minimum_latitude maximum_latitude]
    - vmin = minimum value of the colorbar 
    - vmax = maximum value of the colorbar
    - merv = a 1 x 4 array to know if to label and where the meridians
    -- [0 0 0 0] = no labels
    -- [1 0 0 1]
    - parv = like merv, but for the parallels' labels
    - cmap = colormap to be used
    Output variables:
    - pp = plot object
    """
    lomin = box[0]
    lomax = box[1]
    lamin = box[2]
    lamax = box[3]
    
    map = Basemap(llcrnrlon=lomin,llcrnrlat=lamin,urcrnrlon=lomax,urcrnrlat=lamax,
         resolution='i', projection='merc', lat_0 = (lamin+lamax)/2, lon_0 = (lomin+lomax)/2
                  , ax=ax)
    map.drawcoastlines()

    map.fillcontinents(color='#ddaa66')
    map.drawmeridians(np.arange(-160, 140, 2), labels=merv, size=18)
    map.drawparallels(np.arange(0.5, 70.5, 2), labels=parv, size=18)

    pp = map.scatter(xx, yy, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, latlon=True
                     , cmap = cmap)
    pp.set_clim([vmin, vmax])
    
    return pp

#=======================================================================

def SWOT_filter(input_filename, output_filename, lambd, iter_max, regularization_model, plot, savefig):
        """
        Reads the input data and filters it according to the specified parameters (lambd, iter_max and regularization_model) and saves the output data in a netCDF file
        (output_filename) and plots and saves the output if specified.
        Input variables:
        - input_filename       = 'input_directory' + 'input_filename'
        - output_filename      = 'output_directory' + 'output_filename'
        - lambd                = lambda: regularization parameter (the larger lambda, the more regular is the denoised image).  This parameter needs to be
                                 adapted to the regularization model. 
        - iter_max             = maximum number of iterations
        - regularization_model = 0 if Tikhonov regularization and 1 if second order penalization (regularization of vorticity)
        - plot                 = 'yes' if you want the figure of the non-filtered/filtered SSH to be shown, 'no' if not.
        - savefig              = 'yes' if you want the figure of the non-filtered/filtered SSH to be saved (png format), 'no' if not.
        Output variables:
        - filt_SSH_model = SSH_model filtered (same shape and dimensions as original array) 
        - filt_SSH_obs   = SSH_obs filtered (same shape and dimensions as original array)
        """
        output_file = Dataset(output_filename, 'r+')
	
        ##====================== Load SWOT data ========================

	nc = Dataset(input_filename)
	lon = nc.variables['lon'][:]
	lat = nc.variables['lat'][:]
	tim = nc.variables['time'][:]
	SSH_obs = nc.variables['SSH_obs'][:]
	SSH_model = nc.variables['SSH_model'][:]
	nc.close()
	
	
	##=================Preparing the SWOT data arrays===============
	# 1. Flipping the SWOT array: (for pass 9 which is descending)
	ldif = lon[0,1] - lon[0,0]
	if (ldif<0):
    		lon = np.fliplr(lon)
    		lat = np.fliplr(lat)
    		SSH_model = np.fliplr(SSH_model)
    		SSH_obs = np.fliplr(SSH_obs)
	
	# 2. Masking the SWOT outputs
	SSH_model = np.ma.masked_invalid(SSH_model)
	SSH_obs = np.ma.masked_invalid(SSH_obs)

	# 3. Separate into left and right swath
	
	nhalfswath=np.shape(lon)[1]/2

	lon_left = lon[:,0:nhalfswath]
	lon_right = lon[:,nhalfswath:]
	lat_left = lat[:,0:nhalfswath]
	lat_right = lat[:,nhalfswath:]

	SSH_model_left = SSH_model[:,0:nhalfswath]
	SSH_model_right = SSH_model[:,nhalfswath:]

	SSH_obs_left = SSH_obs[:,0:nhalfswath]
	SSH_obs_right = SSH_obs[:,nhalfswath:]
	
	##=======================Applying the filter====================

	filt_SSH_model_left = filter(SSH_model_left, iter_max, lambd, regularization_model)
	filt_SSH_model_right = filter(SSH_model_right, iter_max, lambd, regularization_model)
	filt_SSH_obs_left = filter(SSH_obs_left, iter_max, lambd, regularization_model)
	filt_SSH_obs_right = filter(SSH_obs_right, iter_max, lambd, regularization_model)

	##===============Concatenating left and right swaths============
	
        filt_SSH_model = np.concatenate((filt_SSH_model_left, filt_SSH_model_right), axis=1)
	filt_SSH_obs = np.concatenate((filt_SSH_obs_left, filt_SSH_obs_right), axis=1)

	##===================Writing output NetCDF file=================
	#The output file has been created earlier (see above). Here we check whether the   variables we want to save already exist in this file, or not. If not, they are created in the NetCDF file. The filtered SSH fields are then saved in these variables.

	variables_output_file = output_file.variables.keys()

	if "SSH_model_filtered" not in variables_output_file:
    		# Creates variable:
    		SSH_model_filtered = output_file.createVariable('SSH_model_filtered', np.float32, ('time', 'x_ac'))
    		SSH_obs_filtered = output_file.createVariable('SSH_obs_filtered', np.float32, ('time', 'x_ac'))
    		# Define units:
    		SSH_model_filtered.units = "m"
    		SSH_obs_filtered.units = "m"
    		# Define long_name:
    		SSH_model_filtered.long_name = "SSH interpolated from model filtered using a first order penalization (Tikhonov regularization) filter"
    		SSH_obs_filtered.long_name = "Observed SSH (SSH_model+errors) filtered using a first order penalization (Tikhonov regularization) filter" 
                # Define filter parameter(s) as attribute:
                SSH_model_filtered.Iter_max = iter_max
		SSH_model_filtered.Lambda = lambd
                SSH_obs_filtered.Iter_max = iter_max
                SSH_obs_filtered.Lambda = lambd

    		print 'Created the filtered variables'
	else:
    		print 'Variables already created'

	appendvar_m = output_file.variables['SSH_model_filtered']
	appendvar_m[:,:] = filt_SSH_model[:,:]
	appendvar_o = output_file.variables['SSH_obs_filtered']
	appendvar_o[:,:] = filt_SSH_obs[:,:]

	print 'Variables updated'

	output_file.close()
        
        ##========================Plotting=========================
        if plot == 'yes':
		# Calculating the absolute difference between SSH_obs and _model:
		diff_nofilt = abs(SSH_obs - SSH_model)

		diff_filt = abs(filt_SSH_obs - filt_SSH_model)
		
                # Checking if pass 9 or 22:
		passnb = input_filename.split('_p')[-1].split('.')[0] 
		
		if passnb == '009':
		 	box = [2., 7., 36., 44.] #lomin, lomax, lamin, lamax
		elif passnb == '022':
			box = [0., 5., 36., 42.5] #lomin, lomax, lamin, lamax
		else:
			print 'problem'
		vmin = -.2
		vmax = 0.2 

		vmin2 = 0.
		vmax2 = 0.1

		plt.figure(figsize=(16, 16))

		gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1]
				       , height_ratios=[.49, .49, .02])

		ax_nf_m = plt.subplot(gs[0,0])
		ax_nf_o = plt.subplot(gs[0,1])
		ax_nf_d = plt.subplot(gs[0,2])
		ax_f_m = plt.subplot(gs[1,0])
		ax_f_o = plt.subplot(gs[1,1])
		ax_f_d = plt.subplot(gs[1,2])
		axC1 = plt.subplot(gs[-1, 0:2])
		axC2 = plt.subplot(gs[-1, -1])

		# No filter:
		pp = plot_map(lon, lat, SSH_model, ax_nf_m, box, vmin, vmax, merv = [0,0,0,0]
			      , parv = [1,0,0,1], cmap='seismic')
		ax_nf_m.set_title('SSH_model \n ', size=20)
		ax_nf_m.set_ylabel('Non-filtered', labelpad=80, size = 20)

		plot_map(lon, lat, SSH_obs, ax_nf_o, box, vmin, vmax, merv = [0,0,0,0]
			 , parv = [0,0,0,0], cmap='seismic')
		ax_nf_o.set_title('SSH_obs \n ', size=20)

		pp2 = plot_map(lon, lat, diff_nofilt, ax_nf_d, box, vmin2, vmax2
			       , merv = [0,0,0,0], parv = [0,0,0,0], cmap='viridis')
		ax_nf_d.set_title('Absolute difference \n (SSH_obs - SSH_model)', size=20)

		# Filtered:
		pp = plot_map(lon, lat, filt_SSH_model, ax_f_m, box, vmin, vmax, merv = [1,0,0,1]
			      , parv = [1,0,0,1], cmap='seismic')
		ax_f_m.set_ylabel('Filtered', labelpad=80, size = 20)

		plot_map(lon, lat, filt_SSH_obs, ax_f_o, box, vmin, vmax, merv = [1,0,0,1]
			 , parv = [1,0,0,1], cmap='seismic')

		pp2 = plot_map(lon, lat, diff_filt, ax_f_d, box, vmin2, vmax2
			       , merv = [1,0,0,1], parv = [1,0,0,1], cmap='viridis')

		cbar1 = plt.colorbar(pp, cax=axC1, orientation='horizontal')
		cbar2 = plt.colorbar(pp2, cax=axC2, orientation='horizontal')
		cbar1.ax.tick_params(labelsize=18)
		cbar2.ax.tick_params(labelsize=18)
		cbar2.set_ticks(np.arange(0., 0.3, 0.1))
                
                if savefig == 'yes':
                    filenamep = output_filename.split('/')[-1].split('.')[0]
                    folder = output_filename.split('/Users')[-1].split(filenamep)[0]
                    savename = '/Users' + folder  + 'figs/' + filenamep + '.png'
                    plt.savefig(str(savename), bbox_inches='tight')
                
                plt.show()
        
        return filt_SSH_model, filt_SSH_obs 
	

