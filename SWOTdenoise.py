# %filter_SWOT_data.py

import numpy as np
from netCDF4 import Dataset
from scipy import ndimage as nd
from scipy.interpolate import RectBivariateSpline
from types import *
import sys


def read_data(filename, *args):
    """Read arrays from netcdf file.
    args arguments are the variables to be read. The number of output arrays must be identical to the number of variables."""
    fid = Dataset(filename)
    output = []
    for entry in args:
        output.append( fid.variables[entry][:] )
    fid.close()
    return tuple(output)


def write_data(filename, ssh_d, lon_d, lat_d, x_ac_d, time_d):
    """Write filtered SSH in output file."""
    
    # Output file name
    rootname = filename.split('.nc')[0]
    filenameout = rootname+'_denoised.nc'
    
    # Read variables (not used before) in input file
    x_al_r = read_data(filename, 'x_al')
    
    # Create output file
    fid = Dataset(filenameout, 'w', format='NETCDF4')
    fid.description = "Filtered SWOT data"
    fid.creator_name = "SWOTdenoise module"  

    # Dimensions
    time = fid.createDimension('time', len(time_d))
    x_ac = fid.createDimension('x_ac', len(x_ac_d))

    # Create variables
    lat = fid.createVariable('lat', 'f8', ('time','x_ac'))
    lat.long_name = "Latitude" 
    lat.units = "degrees_north"
    lat[:] = lat_d
  
    lon = fid.createVariable('lon', 'f8', ('time','x_ac'))
    lon.long_name = "longitude" 
    lon.units = "degrees_east"
    lon[:] = lon_d

    """
    lon_nadir = fid.createVariable('lon_nadir', 'f8', ('time'))
    lon_nadir.long_name = "longitude nadir" 
    lon_nadir.units = "degrees_north"
    lon_nadir = vlon_nadir

    lat_nadir = fid.createVariable('lat_nadir', 'f8', ('time'))
    lat_nadir.long_name = "latitude nadir" 
    lat_nadir.units = "degrees_north"
    lat_nadir[:] = vlat_nadir
    """
       
    vtime = fid.createVariable('time', 'f8', ('time'))
    vtime.long_name = "time from beginning of simulation" 
    vtime.units = "days"
    vtime[:] = time_d

    x_al = fid.createVariable('x_al', 'f8', ('time'))
    x_al.long_name = "Along track distance from the beginning of the pass" 
    x_al.units = "km"
    x_al[:] = x_al_r

    vx_ac = fid.createVariable('x_ac', 'f8', ('x_ac'))
    vx_ac.long_name = "Across track distance from nadir" 
    vx_ac.units = "km"
    vx_ac[:] = x_ac_d

    ssh = fid.createVariable('SSH', 'f8', ('time','x_ac'))
    ssh.long_name = "SSH denoised" 
    ssh.units = "m"
    ssh[:] = ssh_d

    fid.close()  # close the new file
    return filenameout
    
def copy_arrays(*args):
    """numpy-copy arrays.
    *args arguments are the numpy arrays to copy. The number of output arrays must be identical to the number of arguments.
    """
    output = []
    for entry in args:
        #print entry
        output.append( entry.copy() )
    return tuple(output)

def fill_nadir_gap(ssh, lon, lat, x_ac, time, method = 'fill_value'):
    """
    Fill the nadir gap in the middle of SWOT swath.
    Longitude and latitude are interpolated linearly. For SSH, there are two options:
    - method = 'fill_value': the gap is filled with the fill value of SSH masked array;
    - method = 'interp': the gap is filled with a 2D, linear interpolation."""
    
    # Extend x_ac, positions of SWOT pixels across-track
    nhsw     = len(x_ac)/2                                # number of pixels in half a swath
    step     = abs(x_ac[nhsw+1]-x_ac[nhsw])               # x_ac step, constant
    ins = np.arange(x_ac[nhsw-1], x_ac[nhsw], step)[1:]   # sequence to be inserted
    x_ac_f = np.insert(x_ac, nhsw, ins)                   # insertion
    nins = len(ins)                                       # length of inserted sequence
    
    # 2D arrays: lon, lat. Interpolation of regular grids.
    lon_f = RectBivariateSpline(time, x_ac, lon)(time, x_ac_f)
    lat_f = RectBivariateSpline(time, x_ac, lat)(time, x_ac_f)
    
    # SSH: interpolate or insert array of fill values, and preserve masked array characteristics
    if method == 'interp':
        ssh_f = np.ma.masked_values( RectBivariateSpline(time, x_ac, ssh)(time, x_ac_f), ssh.fill_value )
    else:
        ins_ssh = np.full( ( nins, len(time) ), ssh.fill_value, dtype=None )
        ssh_f = np.ma.masked_values( np.insert( ssh, nhsw, ins_ssh, axis=1 ), ssh.fill_value )

    return ssh_f, lon_f, lat_f, x_ac_f


def gaussian_filter(ssh, lon, lat, x_ac, time, lambd, inpainting = False):
    """Apply Gaussian filter on ssh. 'lambd' is the standard deviation of the Gaussian, i.e. the strength of the filter.
    """
    ssh_f, lon_f, lat_f, x_ac_f = fill_nadir_gap(ssh, lon, lat, x_ac, time)
    ssh_d = gaussian_with_nans(ssh_f, lambd)
    
    if inpainting is True:
        ssh_tmp, _, _, _ = fill_nadir_gap(ssh, lon, lat, x_ac, time, method='interp')
        ssh_d = np.ma.array(ssh_d, mask = ssh_tmp.mask)
    else:
        ssh_d = np.ma.array(ssh_d, mask = ssh_f.mask)
    
    return ssh_d, lon_f, lat_f, x_ac_f, time

def gaussian_with_nans(u, sigma):
    """
    Method to implement Gaussian filter on an image with gaps.
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    """
    assert np.ma.any(u.mask), 'u must be a masked array'
    mask = np.flatnonzero(u.mask)
    
    v = u.data.copy()
    v.flat[mask] = 0
    v[:] = nd.gaussian_filter(v ,sigma=sigma)

    w = np.ones_like(u.data)
    w.flat[mask] = 0
    w[:] = nd.gaussian_filter(w, sigma=sigma)
    
    #v = np.ma.array(v, mask=u.mask)
    w = np.clip( w, 1.e-8, 1.)

    return v/w
 
def boxcar_filter(ssh, lon, lat, x_ac, time, lambd, inpainting = False):
    """Apply boxcar filter on ssh. 'lambd' is the footprint (width) of the box.
    """
    ssh_f, lon_f, lat_f, x_ac_f = fill_nadir_gap(ssh, lon, lat, x_ac, time)
    ssh_d = boxcar_with_nans(ssh_f, lambd)
    
    if inpainting is True:
        ssh_tmp, _, _, _ = fill_nadir_gap(ssh, lon, lat, x_ac, time, method='interp')
        ssh_d = np.ma.array(ssh_d, mask = ssh_tmp.mask)
    else:
        ssh_d = np.ma.array(ssh_d, mask = ssh_f.mask)
    
    return ssh_d, lon_f, lat_f, x_ac_f, time

def boxcar_with_nans(u, size):
    """
    Method to implement boxcar filter on an image with gaps.
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    """
    assert np.ma.any(u.mask), 'u must be a masked array'
    mask = np.flatnonzero(u.mask)
    
    v = u.data.copy()
    v.flat[mask] = 0
    v[:] = nd.generic_filter(v ,function=np.nanmean, size = size)

    w = np.ones_like(u.data)
    w.flat[mask] = 0
    w[:] = nd.generic_filter(w, function=np.nanmean, size = size)
    
    #v = np.ma.array(v, mask=u.mask)
    w = np.clip( w, 1.e-8, 1.)

    return v/w
 


def input_error():
    print "You must provide a SWOT file name OR SSH, lon, lat, x_ac and time arrays. SSH must be a masked array."
    sys.exit()


################################################################
# Main function:

def SWOTdenoise(*args, **kwargs):
    # , method, parameter, inpainting='no',
    """
    Driver function.  
    """
    
    ################################################################
    # 1. Read function arguments
    
    # 1.1. Input data
    
    file_input = len(args) == 1
    if file_input:
        if type(args[0]) is not str: input_error()  
        filename = args[0]
        swotfile = filename.split('/')[-1]
        swotdir = filename.split(swotfile)[0]
        listvar = 'SSH_obs', 'lon', 'lat', 'x_ac', 'time'
        ssh, lon, lat, x_ac, time = read_data(filename, *listvar)      
    else:
        ssh         = kwargs.get('ssh', None)
        lon         = kwargs.get('lon', None)
        lat         = kwargs.get('lat', None)
        x_ac        = kwargs.get('x_ac', None)
        time        = kwargs.get('time', None)
        if any( ( isinstance(ssh, NoneType), isinstance(lon, NoneType), isinstance(lat, NoneType), \
                  isinstance(x_ac, NoneType), isinstance(time, NoneType) ) ):
            input_error()
           
    # 1.2. Denoising method
           
    method = kwargs.get('method', 'penalization_order_3')
    lambd = kwargs.get('lambda', 1.5)          # default value to be defined 
    itermax = kwargs.get('itermax', 500)
    epsilon = kwargs.get('epsilon', 1.e-6)
    inpainting = kwargs.get('inpainting', False)  # Only for quadratic penalization methods
    
    # 2. Perform denoising
    
    if method == 'do_nothing':
        ssh_d, lon_d, lat_d, x_ac_d, time_d = copy_arrays(ssh, lon, lat, x_ac, time)
        
    if method == 'boxcar':
        ssh_d, lon_d, lat_d, x_ac_d, time_d = boxcar_filter(ssh, lon, lat, x_ac, time, lambd)

    if method == 'gaussian_filter':
        ssh_d, lon_d, lat_d, x_ac_d, time_d = gaussian_filter(ssh, lon, lat, x_ac, time, lambd)

    if method == 'penalization_order_1':
        pass
#        ssh_filtered = tickonov_filter_1(ssh, lon, lat, lambd, itermax, epsilon, inpainting=inpainting)

    if method == 'penalization_order_3':
        pass
#        ssh_filtered = tickonov_filter_3(ssh, lon, lat, lambd, itermax, epsilon, inpainting=inpainting)

    

    # 3. Manage results
    
    if file_input:
        """copy initial file and replace SSH with filtered SSH. To be done."""
        filenameout = write_data(filename, ssh_d, lon_d, lat_d, x_ac_d, time_d)
        print 'Filtered field in ', filenameout
    else:
        if inpainting is None:
            return ssh_d
        else:
            return ssh_d, lon_d, lat_d, x_ac_d, time_d
    
    
    
   