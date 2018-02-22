# %filter_SWOT_data.py

import numpy as np
from netCDF4 import Dataset


def read_data(filename, *args):
    """Read arrays from netcdf file.
    args arguments are the variables to be read. The number of output arrays must be identical to the number of variables."""
    fid = Dataset(filename)
    output = []
    for entry in args:
        #print entry
        output.append( np.array(fid.variables[entry]) )
    fid.close()
    return tuple(output)


def write_data(filename, ssh_d, lon_d, lat_d, x_ac_d):
    """Write filtered SSH in output file."""
    
    # Output file name
    rootname = filename.split('.nc')[0]
    filenameout = rootname+'_denoised.nc'
    
    # Read variables (not used before) in input file
    time_r, x_al_r = read_data(filename, 'time', 'x_al')
    
    # Create output file
    fid = Dataset(filenameout, 'w', format='NETCDF4')
    fid.description = "Filtered SWOT data"
    fid.creator_name = "SWOTdenoise module"  

    # Dimensions
    time = fid.createDimension('time', len(time_r))
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
    vtime[:] = time_r

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
        output.append( np.copy(entry) )
    return tuple(output)

################################################################
# Main function:

def SWOTdenoise(**kwargs):
    # , method, parameter, inpainting='no',
    """
    Driver function.  
    """
    
    ################################################################
    # 1. Read function arguments
    
    # 1.1. Input data
    
    filename = kwargs.get('filename', None)
    
    if filename is not None:
        swotfile = filename.split('/')[-1]
        swotdir = filename.split(swotfile)[0]
        #listvar = ADT_obs_box, lon_box, lat_box, x_ac
        listvar = 'SSH_obs', 'lon', 'lat', 'x_ac'
        ssh, lon, lat, x_ac = read_data(filename, *listvar)      
    else:
        ssh         = kwargs.get('data', None)
        lon         = kwargs.get('lon', None)
        lat         = kwargs.get('lat', None)
        x_ac        = kwargs.get('x_ac', None)
        #if any( (ssh.any(), lon.any(), lat, x_ac) ) is None:
        if any( (isinstance(ssh, np.ndarray), isinstance(lon, np.ndarray), \
                 isinstance(lat, np.ndarray), isinstance(x_ac, np.ndarray) ) ) is None: 
            print "You must provide a SWOT file name or SSH, lon, lat, and x_ac arrays"
            sys.exit()    
        
        
    # 1.2. Denoising method
           
    method = kwargs.get('method', 'penalization_order_3')
    lambd = kwargs.get('lambda', 1.5)          # default value to be defined 
    itermax = kwargs.get('itermax', 500)
    epsilon = kwargs.get('epsilon', 1.e-6)
    inpainting = kwargs.get('inpainting', None)  # Only for quadratic penalization methods
    
    # 2. Perform filtering
    
    if method == 'boxcar':
        pass
#        ssh_filtered = boxcar_filter(ssh, lon, lat, lambda)

    if method == 'gaussian_filter':
        pass
#        ssh_filtered = gaussian_filter(ssh, lon, lat, lambda)

    if method == 'penalization_order_1':
        pass
#        ssh_filtered = tickonov_filter_1(ssh, lon, lat, lambda, itermax, epsilon, inpainting=inpainting)

    if method == 'penalization_order_3':
        pass
#        ssh_filtered = tickonov_filter_3(ssh, lon, lat, lambda, itermax, epsilon, inpainting=inpainting)
    ssh_d, lon_d, lat_d, x_ac_d = copy_arrays(ssh, lon, lat, x_ac)
    

    # 3. Manage results
    
    if filename is not None:
        "copy initial file and replace SSH with filtered SSH. To be done."
        filenameout = write_data(filename, ssh_d, lon_d, lat_d, x_ac_d)
        print 'Filtered field in ', filenameout
    else:
        if inpainting is None:
            return ssh_d
        else:
            return ssh_d, lon_d, lat_d, x_ac_d
    
    
    
   