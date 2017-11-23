# Necessary modules:
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import numpy as np
import glob
from collections import Counter
from netCDF4 import Dataset
import GriddedData

#from math import sqrt

# Necessary functions:
## N. Papadakis:
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

def laplacian_np(u):
    """
    Calculates the laplacian of u using the divergence and gradient functions and gives 
    as output Ml.
    """
    Ml = div(gradi(u), gradj(u));
    return Ml

## griddedData
def masked(array):
    """
    """
    m1 = np.ma.masked_invalid(array)
    m2 = np.ma.masked_equal(m1, 0.)
    return m2

def rel_vort_griddedData(lat, lon, data):
    """
    Calculates the relative vorticity (1/s).  Firstly the laplacian is calculated, and then used to calculate 
    relative vorticity.
    Inputs:
    -
    Output:
    -
    """
    mgrd = GriddedData.grid2D(navlat=lat, navlon=lon)
    corio = GriddedData.corio(mgrd)
    lap = mgrd.div(mgrd.grad(data))
    lap = np.ma.masked_greater(masked(lap),10.)
    
    gx_corio, gy_corio = mgrd.grad(corio)
    beta = gy_corio
    gx_data, gy_data = mgrd.grad(data)
    
    minus_factor = ((grav * beta) / corio**2) * gy_data
    
    rel_vort = (grav * lap / corio ) - minus_factor
    
    return rel_vort

def laplacian_griddedData(lat, lon, data):
    mgrd = GriddedData.grid2D(navlat=lat, navlon=lon)
    lap = mgrd.div(mgrd.grad(data))
    lap = np.ma.masked_greater(masked(lap),10.)
    
    return lap

# Necessary parameters:
grav = 9.81

#########################################################################################
class swot_pass():
    def __init__(self, datadir, namfiles, namdescr):
        self.datadir = datadir
        self.namfiles = namfiles
        self.namdescr = namdescr
        fdescr = open(self.namdescr)
        flines = fdescr.readlines()
        fdescr.close()
        exec('self.' + flines[5]) # corresponds to delta_ac [km]
        exec('self.' + flines[6]) # corresponds to delta_al [km]
        exec('self.' + flines[16]) # corresponds to halfgap size [km] 
        
        self.dx = self.delta_ac * 1000 # so in m
        self.dy = self.delta_al * 1000 # so in m
        self.grid_size = self.delta_ac * 1000 # so in m # as regular SWOT grid, can use either delta_ac or _al
        
    def get_list_of_swotfiles(self):
        """
        Obtain a list of all the available SWOT files
        Function(s) used in:
        - get_list_of_cycles
        - get_list_of_passes
        """
        return glob.glob(self.datadir + self.namfiles + '*.nc')

    def get_list_of_cycles(self):
        """
        Obtain a list of all the possible cycles
        """
        swotfiles = self.get_list_of_swotfiles()
        swotcycles = [s.split('_c')[-1].split('_')[0] for s in swotfiles]
        npcycles = np.array(swotcycles,dtype=int)
        # To get back the cycle number as a list instead of array:    
        dictcount = Counter(npcycles)
        return dictcount.keys()

    def get_list_of_passes(self):
        """
        Obtain a list of all the possible passes
        """
        swotfiles = self.get_list_of_swotfiles()
        swotpasses = [s.split('_p')[-1].split('.')[0] for s in swotfiles]
        nppasses = np.array(swotpasses,dtype=int)
        # To get back the cycle number as a list instead of array:
        dictcount = Counter(nppasses)
        passes = dictcount.keys()
        passes.sort()
        return passes
    
    def read_data_ncfile(self, swotfile):
        """
        Read SWOT data from netcdf file for a particular:
        -swotfile: input file (directory + filename), choose an index from swotfiles
        """
        self.myfile = swotfile
        nc = Dataset(self.myfile)
        self.lon = nc.variables['lon'][:]
        self.lat = nc.variables['lat'][:]
        #self.time = nc.variables['time'][:]
        self.SSH_model = nc.variables['SSH_model'][:]
        self.SSH_obs = nc.variables['SSH_obs'][:]
        self.x_ac = nc.variables['x_ac'][:]
        nc.close()    
        
        self.nhalfswath = np.shape(self.lon)[1]/2
        
	# Doing the transpose of the data:
        # np.tranpose applied to the data as it's originally organized yy, xx (along track, across track) and 
        # for it to be xx, yy and more natural to do the gradient and divergnece calculations
    
        self.lon = np.transpose(self.lon)
        self.lat = np.transpose(self.lat)
        self.SSH_model = np.transpose(self.SSH_model)
        self.SSH_obs = np.transpose(self.SSH_obs)    
        
    def read_data_ncfile_box(self, swotfile):
        """
        Read SWOT data from netcdf file for a particular:
        -swotfile: input file (directory + filename), choose an index from swotfiles
        """
        self.myfile = swotfile
        nc = Dataset(self.myfile)
        self.lon = nc.variables['lon_box'][:]
        self.lat = nc.variables['lat_box'][:]
        #self.time = nc.variables['time'][:]
        self.SSH_obs = nc.variables['ADT_obs_box'][:]
        self.SSH_model = nc.variables['ADT_model_box'][:]
        self.x_ac = nc.variables['x_ac'][:]
        nc.close()    
        
        self.nhalfswath = np.shape(self.lon)[1]/2
        
	# Doing the transpose of the data:
        # np.tranpose applied to the data as it's originally organized yy, xx (along track, across track) and 
        # for it to be xx, yy and more natural to do the gradient and divergnece calculations
    
        self.lon = np.transpose(self.lon)
        self.lat = np.transpose(self.lat)
        self.SSH_model = np.transpose(self.SSH_model)
        self.SSH_obs = np.transpose(self.SSH_obs)    
        
        #####
        
        SSH_obs_left, SSH_obs_right = self.separate_swaths(self.SSH_obs)
        xx_left, yy_left = np.where(~SSH_obs_left.mask)
        xx_right, yy_right = np.where(~SSH_obs_right.mask)

        self.left_gap_indices = np.unique(xx_left)
        self.right_gap_indices = np.unique(xx_right)

    def read_data_ncfile_filtered(self, swotfile):
        """
        Read SWOT data from netcdf file for a particular:
        -swotfile: input file (directory + filename), choose an index from swotfiles
        """
        self.myfile = swotfile
        nc = Dataset(self.myfile)
        self.lon = nc.variables['lon'][:]
        self.lat = nc.variables['lat'][:]
        #self.time = nc.variables['time'][:]
        self.SSH_model = nc.variables['SSH_model'][:]
        self.SSH_obs = nc.variables['SSH_obs'][:]
        self.SSH_modelf = nc.variables['SSH_model_filtered'][:]
        self.SSH_obsf = nc.variables['SSH_obs_filtered'][:]
        self.x_ac = nc.variables['x_ac'][:]
        nc.close()    
        
        self.nhalfswath = np.shape(self.lon)[1]/2
        
	# Doing the transpose of the data:
        # np.tranpose applied to the data as it's originally organized yy, xx (along track, across track) and 
        # for it to be xx, yy and more natural to do the gradient and divergnece calculations
    
        self.lon = np.transpose(self.lon)
        self.lat = np.transpose(self.lat)
        self.SSH_model = np.transpose(self.SSH_model)
        self.SSH_obs = np.transpose(self.SSH_obs)    
        self.SSH_modelf = np.transpose(self.SSH_modelf)
        self.SSH_obsf = np.transpose(self.SSH_obsf)  

    def read_data_ncfile_box_filtered(self, swotfile):
        """
        Read SWOT data from netcdf file for a particular:
        -swotfile: input file (directory + filename), choose an index from swotfiles
        """
        self.myfile = swotfile
        nc = Dataset(self.myfile)
        self.lon = nc.variables['lon_box'][:]
        self.lat = nc.variables['lat_box'][:]
        #self.time = nc.variables['time'][:]
        self.SSH_obs = nc.variables['ADT_obs_box'][:]
        self.SSH_model = nc.variables['ADT_model_box'][:]
        self.SSH_obsf = nc.variables['SSH_obs_filtered'][:]
        self.SSH_modelf = nc.variables['SSH_model_filtered'][:]
        #self.x_ac = nc.variables['x_ac'][:]
        nc.close()    
        
        self.nhalfswath = np.shape(self.lon)[1]/2
        
	# Doing the transpose of the data:
        # np.tranpose applied to the data as it's originally organized yy, xx (along track, across track) and 
        # for it to be xx, yy and more natural to do the gradient and divergnece calculations
    
        self.lon = np.transpose(self.lon)
        self.lat = np.transpose(self.lat)
        self.SSH_model = np.transpose(self.SSH_model)
        self.SSH_obs = np.transpose(self.SSH_obs)    
        self.SSH_modelf = np.transpose(self.SSH_modelf)
        self.SSH_obsf = np.transpose(self.SSH_obsf)  
        
        # INDICES
        self.left_gap_indices = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        self.right_gap_indices = [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    def read_data_ncnp(self, ncycle, npass):
        """
        Read SWOT data from netcdf file for a particular:
        -ncycle: Cycle number (run get_list_of_cycles)
        -npass: Pass number (run get_list_of_passes)
        """
        self.myfile = self.datadir + self.namfiles \
                + 'c' + str(ncycle).zfill(2) \
                + '_p' + str(npass).zfill(3) + '.nc'
        nc = Dataset(self.myfile)
        self.lon = nc.variables['lon_box'][:]
        self.lat = nc.variables['lat_box'][:]
        #self.time = nc.variables['time'][:]
        self.SSH_obs = nc.variables['ADT_obs_box'][:]
        self.SSH_model = nc.variables['ADT_model_box'][:]
        self.SSH_obsf = nc.variables['SSH_obs_filtered'][:]
        self.SSH_modelf = nc.variables['SSH_model_filtered'][:]
        nc.close()    
	
        # Doing the transpose of the data:
        # np.tranpose applied to the data as it's originally organized yy, xx (along track, across track) and 
        # for it to be xx, yy and more natural to do the gradient and divergnece calculations
        
        self.lon = np.transpose(self.lon)
        self.lat = np.transpose(self.lat)
        self.SSH_model = np.transpose(self.SSH_model)
        self.SSH_obs = np.transpose(self.SSH_obs)	
	self.SSH_modelf = np.transpose(self.SSH_modelf)
        self.SSH_obsf = np.transpose(self.SSH_obsf)         

    def separate_swaths(self, var):
        """
        Separating into left and right swaths a particular variable, var, ignoring the nadir.
        - var: variable to separate
        Function(s) used in:
        - gradient_norm_method1
        - velocity_norm
        - var_on_ugrid
        - var_on_vgrid
        - var_on_fgrid
        - geostrophic_velocities_method1
        - geostrophic_velocities_method2
        - geostrophic_velocities_method3
        - rel_vorticity_method2
        - rel_vorticity_method3
        - rossby_nb_method2
        - rossby_nb_method3
        """
    
	ii, jj = var.shape
        
        if ii % 2 == 0:

            var_left = var[0:self.nhalfswath,:] #self.nhalfswath = ii/2 
            #--> only possible for SSH, if for vel, need not same nhalfswath when calculated via method 1
            var_right = var[self.nhalfswath::,:]
        
        else: # I assume it's because it includes nadir
            var_left = var[0:self.nhalfswath,:]
            var_right = var[self.nhalfswath+1::,:]
        
        return var_left, var_right
    
    def separate_swaths_nogap(self, var):
        
        var_left, var_right = self.separate_swaths(var)
         
        #--> only possible for SSH, if for vel, need not same nhalfswath when calculated via method 1

        var_left_nogap = var_left[self.left_gap_indices, :]
        var_right_nogap = var_right[self.right_gap_indices, :]
        
        return var_left_nogap, var_right_nogap

    def gradient_method1(self, var):
        """
        Calculating the gradient using method 1.
        Input(s):
        -var: variable (SSH [m]) to calculate gradient from in format var(lon, lat).
        Output(s):
        - SSH gradient [no units]:
            - dvar_dx: derivative (gradient) in the x-direction (along axis 0). Dimensions: (dimlon-1, dimlat)
            - dvar_dy: derivative (gradient) in the x-direction (along axis 0). Dimensions: (dimlon, dimlat-1)
        Functions usesd in:
        - gradient_norm_method1
        """
        
        dvar_x = np.diff(var, axis=0)
        dvar_y = np.diff(var, axis=1)

        dvar_dx = dvar_x/self.dx
        dvar_dy = np.transpose(np.transpose(dvar_y)/self.dy)
        
        return dvar_dx, dvar_dy
    
    def gradient_norm_method1(self, var):
        """
        Calculating and printing the norm of the gradient using method 1.
        Input(s):
        -var: variable to calculate norm of gradient from
        Output(s):
        -
        """
        var_left, var_right = self.separate_swaths(var)
        dvar_dxL, dvar_dyL = self.gradient_method1(var_left)
        dvar_dxR, dvar_dyR = self.gradient_method1(var_right)
        
        var_normgradL = np.nansum(dvar_dyL**2) + np.nansum(dvar_dxL**2)
        var_normgradR = np.nansum(dvar_dyR**2) + np.nansum(dvar_dxR**2)

        var_normgrad = var_normgradL + var_normgradR

        return var_normgrad
    
    def velocity_norm(self, var):
        """
        Calculating the velocity norm using method 2:
        Input(s):
        -var: variable from which it is calculated
        Output(s):
        -
        Functions used in:
        - geostrophic_velocities_method2
        """
    
        var_left, var_right = self.separate_swaths(var)
        var_velocity_norm_left = gradi(var_left)**2 + gradj(var_left)**2
        var_velocity_norm_right = gradi(var_right)**2 + gradj(var_right)**2

        ### Joining the left and right swaths after the calculation for an easier plotting syntax:
        var_velocity_norm = np.concatenate((var_velocity_norm_left, var_velocity_norm_right), axis=0)
        return var_velocity_norm
    
    def var_on_ugrid(self, var, conc):
        """
        Calculates the input variable (var), like for e.g., longitude or latitude on the u-grid:
        (u-grid: )
        Input(s):
        - var: variable to calculate on the u-grid
        - conc: option whether to concatenate ('yes') or not ('no') the left and right swath
        Output(s):
        -
        Functions used in:
        - geostrophic_velocities_method1
        """
        var_left, var_right = self.separate_swaths(var)

        xx, yy = var_left.shape # left and right same shape
        var_u_left = np.zeros((xx, yy-1))
        var_u_right = np.zeros((xx, yy-1))
        for jj in xrange(0, yy-1):
            var_u_left[:, jj] = (var_left[:, jj] + var_left[:, jj+1])/2
            var_u_right[:, jj] = (var_right[:, jj] + var_right[:, jj+1])/2
        if  conc=='no':
            return var_u_left, var_u_right
        elif  conc=='yes':
            var_u = np.concatenate((var_u_left, var_u_right), axis=0)
            return var_u
    
    def var_on_vgrid(self, var, conc):
        """
        Calculates the input variable (var), like for e.g., longitude or latitude on the v-grid:
        (v-grid: )
        Input(s):
        - var: variable to calculate on the v-grid
        - conc: option whether to concatenate ('yes') or not ('no') the left and right swath
        Output(s):
        -
        Functions used in:
        - geostrophic_velocities_method1
        """
        var_left, var_right = self.separate_swaths(var)
        
        xx, yy = var_left.shape # left and right same shape
        var_v_left = np.zeros((xx-1, yy))
        var_v_right = np.zeros((xx-1, yy))
        for ii in xrange(0, xx-1):
            var_v_left[ii, :] = (var_left[ii, :] + var_left[ii+1, :])/2
            var_v_right[ii, :] = (var_right[ii, :] + var_right[ii+1, :])/2
        if  conc=='no':
            return var_v_left, var_v_right
        elif  conc=='yes':
            var_v = np.concatenate((var_v_left, var_v_right), axis=0)
            return var_v
    
    def var_on_fgrid(self, var, conc):
        """
        Calculating lon and lat for the left and right swath on the f grid:
        (f-grid: )
        - var: variable to calculate on the f-grid
        - conc: option whether to concatenate ('yes') or not ('no') the left and right swath
        """
        var_left, var_right = self.separate_swaths(var)

        xx, yy = var_left.shape # left and right same shape
        var_f_left = np.zeros((xx-1, yy-1))
        var_f_right = np.zeros((xx-1, yy-1))
        
        for jj in xrange(0, yy-1):
            for ii in xrange(0, xx-1):
                var_f_left[ii, jj] = (var_left[ii, jj] + var_left[ii+1, jj+1])/2
                var_f_right[ii, jj] = (var_right[ii, jj] + var_right[ii+1, jj+1])/2
        if  conc=='no':
            return var_f_left, var_f_right
        elif  conc=='yes':
            var_f = np.concatenate((var_f_left, var_f_right), axis=0)
            return var_f
    
    def coriolis_parameter(self, lat_g_left, lat_g_right):
        """
        Calculating the coriolis parameter for the left and right swath on the u, v and f (abs) grids:
        for corio_abs give simply lat
        - lat_g_left: left swath latitude on the grid f wants to obtained on
        - lat_g_right: likewise for right swath
        Functions used in:
        - geostrophic_velocities_method1
        - geostrophic_velocities_method2
        - rel_vorticty_method2
        - rossby_nb_method2
        """
        corio_left = 2 * ((2*np.pi)/86400) * np.sin(np.deg2rad(lat_g_left))
        corio_right = 2 * ((2*np.pi)/86400) * np.sin(np.deg2rad(lat_g_right))
        return corio_left, corio_right
    
    def geostrophic_velocities_method1(self, var):
        """
	Calculating geostrophic velocities using method 1:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s): (dims: x_ac, x_al???)
        - u_g: zonal component of the geostrophic velocity [m/s] (dims: (x_ac-1, x_al?))
        - v_g: meridional component of the geostrophic velocity [m/s] (dims: (lon-1, lat)) 
        - abs_g: absolute geostrophic velocity [m/s]
        Function used in:
        - rel_vorticity_method1        
	"""
        
        var_left, var_right = self.separate_swaths(var)

        # U: (zonal component)
        lat_u_left, lat_u_right = self.var_on_ugrid(self.lat, conc='no')
        corio_left_u, corio_right_u = self.coriolis_parameter(lat_u_left, lat_u_right)
        ddu_var_left = np.diff(var_left, axis=1)
        ug_var_left = (-grav/corio_left_u)*(ddu_var_left/self.dy)
        
        ddu_var_right = np.diff(var_right, axis=1)
        ug_var_right = (-grav/corio_right_u)*(ddu_var_right/self.dy)
        
        u_g = np.concatenate((ug_var_left, ug_var_right), axis=0)
        
        # V: (meridional component)
        lat_v_left, lat_v_right = self.var_on_vgrid(self.lat, conc='no')
        corio_left_v, corio_right_v = self.coriolis_parameter(lat_v_left, lat_v_right)
        
        ddv_var_left = np.diff(var_left, axis=0)
        vg_var_left = (grav/corio_left_v)*(ddv_var_left/self.dx)
        
        ddv_var_right = np.diff(var_right, axis=0)
        vg_var_right = (grav/corio_right_v)*(ddv_var_right/self.dx)
        
        v_g = np.concatenate((vg_var_left, vg_var_right), axis=0)
        
        # Absolute velocity: 
        # (one dimension is ignored for ug and vg to make them the same size)
        abs_vel_var_left = np.sqrt((ug_var_left[0:-1,:]**2) + (vg_var_left[:,0:-1]**2))
        abs_vel_var_right = np.sqrt((ug_var_right[0:-1,:]**2) + (vg_var_right[:,0:-1]**2))
       
        abs_g = np.concatenate((abs_vel_var_left, abs_vel_var_right), axis=0)
        
        return u_g, v_g, abs_g
    
    def geostrophic_velocities_method2(self, var):
        """
        Calculates absolute geostrophic velocity only, using method 2:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        """
        lat_left, lat_right = self.separate_swaths(self.lat)
        corio_left, corio_right = self.coriolis_parameter(lat_left, lat_right)
        
        var_left, var_right = self.separate_swaths(var)
        
        var_abs_g_left = np.sqrt(
            (((grav/corio_left)*(1./self.grid_size))**2)*self.velocity_norm(var_left))
        var_abs_g_right = np.sqrt(
            (((grav/corio_right)*(1./self.grid_size))**2)*self.velocity_norm(var_right))
        
        var_abs_g = np.concatenate((var_abs_g_left, var_abs_g_right), axis=0)
        
        return var_abs_g

    def geostrophic_velocities_method3(self, var):
        """
        Calculates absolute geostrophic velocity only, using method 3 (gap ignored):
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        Function used in:
        - abs_geostrophic_velocity_method3
        """
        lon_left, lon_right = self.separate_swaths(self.lon)
        lat_left, lat_right = self.separate_swaths(self.lat)
        
        var_left, var_right = self.separate_swaths(var)

        mgrd_left = GriddedData.grid2D(navlat=lat_left, navlon=lon_left)
        mgrd_right = GriddedData.grid2D(navlat=lat_right, navlon=lon_right)
        corio_left = GriddedData.corio(mgrd_left)
        corio_right = GriddedData.corio(mgrd_right)
        gx_left, gy_left = mgrd_left.grad(var_left)
        gx_right, gy_right = mgrd_right.grad(var_right)
        
        ug_left = (-grav/corio_left) * gy_left
        vg_left = (+grav/corio_left) * gx_left
        ug_right = (-grav/corio_right) * gy_right
        vg_right = (+grav/corio_right) * gx_right
        
        return ug_left, ug_right, vg_left, vg_right
    
    def geostrophic_velocities_method3_nogap(self, var):
        """
        Calculates absolute geostrophic velocity only, using method 3 (gap ignored):
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        Function used in:
        - abs_geostrophic_velocity_method3
        """
        lon_left, lon_right = self.separate_swaths_nogap(self.lon)
        lat_left, lat_right = self.separate_swaths_nogap(self.lat)
        
        var_left, var_right = self.separate_swaths_nogap(var)

        mgrd_left = GriddedData.grid2D(navlat=lat_left, navlon=lon_left)
        mgrd_right = GriddedData.grid2D(navlat=lat_right, navlon=lon_right)
        corio_left = GriddedData.corio(mgrd_left)
        corio_right = GriddedData.corio(mgrd_right)
        gx_left, gy_left = mgrd_left.grad(var_left)
        gx_right, gy_right = mgrd_right.grad(var_right)
        
        ug_left = (-grav/corio_left) * gy_left
        vg_left = (+grav/corio_left) * gx_left
        ug_right = (-grav/corio_right) * gy_right
        vg_right = (+grav/corio_right) * gx_right
        
        return ug_left, ug_right, vg_left, vg_right
    
    def abs_geostrophic_velocity_method3(self, var):
        """
        Calculates absolute geostrophic velocity only, using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        """
        
        ug_left, ug_right, vg_left, vg_right = self.geostrophic_velocities_method3(var)
        
        var_abs_g_left = np.sqrt((ug_left**2) + (vg_left**2))
        var_abs_g_right = np.sqrt((ug_right**2) + (vg_right**2))
        
        var_abs_g = np.concatenate((var_abs_g_left, var_abs_g_right), axis=0)
        
        return var_abs_g

    def abs_geostrophic_velocity_method3_nogap(self, var):
        """
        Calculates absolute geostrophic velocity only, using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        The left and right swath absolute geostrophic velocities [m/s]
        -var_abs_g_left
        -var_abs_g_right
        """
        
        ug_left, ug_right, vg_left, vg_right = self.geostrophic_velocities_method3_nogap(var)
        
        var_abs_g_left = np.sqrt((ug_left**2) + (vg_left**2))
        var_abs_g_right = np.sqrt((ug_right**2) + (vg_right**2))
        
        #var_abs_g = np.concatenate((var_abs_g_left, var_abs_g_right), axis=0)
        
        return var_abs_g_left, var_abs_g_right
    
    def total_kinetic_energy_method3(self, var):
        """
        Calculates the Total Kinetic Energy (TKE) from absolute geostrophic velocities, using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        """
        
        ug_left, ug_right, vg_left, vg_right = self.geostrophic_velocities_method3(var)
        
        TKE_left = 0.5 * ((ug_left**2) + (vg_left**2))
        TKE_right = 0.5 * ((ug_right**2) + (vg_right**2))
        
        TKE = np.concatenate((TKE_left, TKE_right), axis=0)
        
        return TKE
    
    def laplacian_m3(self, var): 
        """
        Calculates the laplacian (divided by grid size) via GriddedData.
        -var: variable from which it is calculated (SSH)
        """
        lon_left, lon_right = self.separate_swaths(self.lon)
        lat_left, lat_right = self.separate_swaths(self.lat)
        
        var_left, var_right = self.separate_swaths(var)
        
        lap_m3_left = laplacian_griddedData(lat_left, lon_left, var_left)
        lap_m3_right = laplacian_griddedData(lat_right, lon_right, var_right)    
        
        lap_m3 = np.concatenate((lap_m3_left, lap_m3_right), axis=0)
        
        return lap_m3
    
    def laplacian_m3_nogap(self, var):
        """
        """
	lon_left, lon_right = self.separate_swaths_nogap(self.lon)
        lat_left, lat_right = self.separate_swaths_nogap(self.lat)
    
        var_left, var_right = self.separate_swaths_nogap(var)
    
        lap_m3_left = laplacian_griddedData(lat_left, lon_left, var_left)
        lap_m3_right = laplacian_griddedData(lat_right, lon_right, var_right)    
    
        #lap_m3 = np.concatenate((lap_m3_left, lap_m3_right), axis=0)
	return lap_m3_left, lap_m3_right

    def rel_vorticity_method2(self, var):
        """
        Calculates the relative vorticity using method 2.  Firstly the laplacian is calculated:
            Laplacian = div(gradx, grady)
        Then, the laplacian is used to calculate relative vorticity.
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        Functions used in:
        - rossby_nb_method2
        """
        var_left, var_right = self.separate_swaths(var)
        var_laplacian_left = laplacian_np(var_left)
        var_laplacian_right = laplacian_np(var_right)
        
        lat_left, lat_right = self.separate_swaths(self.lat)
        corio_left, corio_right = self.coriolis_parameter(lat_left, lat_right)
        
        beta_left = gradj(corio_left) / self.dy
        factor_left = grav / (corio_left*(self.grid_size**2))
        minuspart_model_left = (grav / (corio_left**2)) * beta_left * (gradj(var_left)/self.dy)

        beta_right = gradj(corio_right) / self.dy
        factor_right = grav / (corio_right*(self.grid_size**2))
        minuspart_model_right = (grav / (corio_right**2)) * beta_right * (gradj(var_right)/self.dy)
        
        rel_vort_var_left = (factor_left * var_laplacian_left) - minuspart_model_left
        rel_vort_var_right = (factor_right * var_laplacian_right) - minuspart_model_right
        
        rel_vort_var = np.concatenate((rel_vort_var_left, rel_vort_var_right), axis=0)
        
        return rel_vort_var
    
    def rel_vorticity_method3(self, var): 
        """
        Calculates relative vorticity using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        Functions used in:
        - rossby_nb_method3
        """
        lon_left, lon_right = self.separate_swaths(self.lon)
        lat_left, lat_right = self.separate_swaths(self.lat)
        
        var_left, var_right = self.separate_swaths(var)
        
        rel_vort_var_left = rel_vort_griddedData(lat_left, lon_left, var_left)
        rel_vort_var_right = rel_vort_griddedData(lat_right, lon_right, var_right)    
        
        rel_vort_var = np.concatenate((rel_vort_var_left, rel_vort_var_right), axis=0)
        
        return rel_vort_var
    
    def rel_vorticity_method3_nogap(self, var):
        """
        Calculates relative vorticity using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        Functions used in:
        - rossby_nb_method3
        """
        lon_left, lon_right = self.separate_swaths_nogap(self.lon)
        lat_left, lat_right = self.separate_swaths_nogap(self.lat)

        var_left, var_right = self.separate_swaths_nogap(var)

        rel_vort_var_left = rel_vort_griddedData(lat_left, lon_left, var_left)
        rel_vort_var_right = rel_vort_griddedData(lat_right, lon_right, var_right)

        #rel_vort_var = np.concatenate((rel_vort_var_left, rel_vort_var_right), axis=0)

        return rel_vort_var_left, rel_vort_var_right

    def rossby_nb_method2(self, var):
        """
        Calculates the Rossby number using method 2:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        """
        lat_left, lat_right = self.separate_swaths(self.lat)
        corio_left, corio_right = self.coriolis_parameter(lat_left, lat_right)
        
        corio = np.concatenate((corio_left, corio_right), axis=0)
        
        rossby_nb = self.rel_vorticity_method2(var) / corio
        
        return rossby_nb
    
    def rossby_nb_method3(self, var):
        """
        Calculates the Rossby number using method 3:
        Input(s):
        -var: variable from which it is calculated (SSH in m)
        Output(s):
        -
        """
        lon_left, lon_right = self.separate_swaths(self.lon)
        lat_left, lat_right = self.separate_swaths(self.lat)
                
        mgrd_left = GriddedData.grid2D(navlat=lat_left, navlon=lon_left)
        mgrd_right = GriddedData.grid2D(navlat=lat_right, navlon=lon_right)
        corio_left = GriddedData.corio(mgrd_left)
        corio_right = GriddedData.corio(mgrd_right)
        
        corio = np.concatenate((corio_left, corio_right), axis=0)
        
        rossby_nb = self.rel_vorticity_method3(var) / corio
        
        return rossby_nb
    
    def plot_single_pass(self, lon, lat, var, vmin, vmax, merv, parv, cmap, extend_opt, land, plotstyle='0', levels=0, **options):
        """
        Plots an individual plot of the data using scatter and Basemap.
        Input(s):
        - var = variable to be plotted
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
	- extend_opt = 'both', 'max', or 'min', to extend colorbar
	- land = 0 if you want only the data to be plotted or 1 if you want the land do be plotted on-top.
        - plotsyle: (0 by default)
            -- '0' only scatter
            -- '1' scatter with contour
        - levels: levels of contour. (0 by default) 
            -- 0 if none (or just scatter)
            -- levels = np.arange(min, max, interval)
        - options:
            -- ax = axis (for e.g. if only one plot, not necessary)
        Output(s):
        - pp = plot object
        """
        
        lomin = self.box[0]
        lomax = self.box[1]
        lamin = self.box[2]
        lamax = self.box[3]
        
	if land == 0:
	    if not options: #(i.e., if no axis define, one single plot)
                
                if plotstyle == '1':
                            plt.contour(lon, lat, var, levels, colors='w', linewidth=2) 

                pp = plt.scatter(lon, lat, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
		plt.xlim([lomin, lomax])
        	plt.ylim([lamin, lamax])
		plt.axis('equal')
                pp.set_clim([vmin, vmax])
            	cbar = plt.colorbar(pp)
            	#cbar.ax.tick_params(labelsize=16)
            else:
		ax = options.get("ax")

                if plotstyle == '1':
                            ax.contour(lon, lat, var, levels, colors='w', linewidth=2) 

                pp = ax.scatter(lon, lat, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
		ax.set_xlim([lomin, lomax])
		ax.set_ylim([lamin, lamax])
                pp.set_clim([vmin, vmax])
		plt.axis('equal')

        elif land == 1:

            if not options:
                my_map = Basemap(llcrnrlon=lomin, llcrnrlat=lamin, urcrnrlon=lomax, urcrnrlat=lamax
                          , resolution='i', projection='merc', lat_0 = (lamin+lamax)/2
                          , lon_0 = (lomin+lomax)/2, area_thresh=10)
            else:
                my_map = Basemap(llcrnrlon=lomin, llcrnrlat=lamin, urcrnrlon=lomax, urcrnrlat=lamax
                          , resolution='i', projection='merc', lat_0 = (lamin+lamax)/2
                          , lon_0 = (lomin+lomax)/2, ax=options.get("ax"), area_thresh=10)

            x, y = my_map(lon, lat)

            if plotstyle == '1':
                my_map.contour(x, y, var, levels, colors='w', linewidth=2) 
            
            pp = my_map.scatter(x, y, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
                    
            my_map.drawcoastlines()
            my_map.fillcontinents(color='#ddaa66')#'gray'
            my_map.drawmeridians(np.arange(-160, 140, 2), labels=merv, size=18)
            my_map.drawparallels(np.arange(0.5, 70.5, 2), labels=parv, size=18)

            pp.set_clim([vmin, vmax])

            if not options:
                cbar = plt.colorbar(pp, extend=extend_opt)
                cbar.ax.tick_params(labelsize=16)
            
        return pp
    
    def plot_model_vs_obs(self, var1, var2, vmin, vmax, cmap, extend_opt, land, plotstyle='0', levels=0):
            """
            Plots a 1x2 figure using scatter and basemap, and with a common colorbar showing 
            horizontally underneath the plots.
            Input(s):
            - var1 = variable of the left plot
            - var2 = variable of the right plot
            - vmin = minimum value of the colorbar 
            - vmax = maximum value of the colorbar
            - cmap = colormap to be used
            - extend_opt = 'both', 'max', or 'min', to extend colorbar
	    - land = 0 if you want only the data to be plotted or 1 if you want the land do be plotted on-top.
            - plotsyle: (0 by default)
            -- '0' only scatter
            -- '1' scatter with contour
	    - levels: levels of contour. (0 by default) 
            -- 0 if none (or just scatter)
            -- levels = np.arange(min, max, interval)
	    Output(s):
            - ax1 = left plot axis object
            - ax2 = right plot axis object
            You need to put the titles by using:
            ax1.set_title('title1', size=18)
            ax2.set_title('title2', size=18)
            """
            
            lon = self.lon
            lat = self.lat
               
            lomin = self.box[0]
            lomax = self.box[1]
            lamin = self.box[2]
            lamax = self.box[3]

            gs = gridspec.GridSpec(2, 2, height_ratios=[0.95, .05], width_ratios=[1, 1])

            fig1 = plt.figure(figsize=(16, 10))  # figsize = (width ,height)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])

            axC = plt.subplot(gs[1, :])

	    if land == 0:

                if plotstyle == '1':
        		ax1.contour(lon[0:self.nhalfswath,:], lat[0:self.nhalfswath,:], var1[0:self.nhalfswath,:], levels, colors='w'
                         , linewidth=2)
			ax1.contour(lon[self.nhalfswath::,:], lat[self.nhalfswath::,:], var1[self.nhalfswath::,:], levels, colors='w'
                         , linewidth=2)
 			ax2.contour(lon[0:self.nhalfswath,:], lat[0:self.nhalfswath,:], var2[0:self.nhalfswath,:], levels, colors='w'                                       , linewidth=2)
			ax2.contour(lon[self.nhalfswath::,:], lat[self.nhalfswath::,:], var2[self.nhalfswath::,:], levels, colors='w'                                       , linewidth=2)

                c1 = ax1.pcolor(lon[0:self.nhalfswath,:], lat[0:self.nhalfswath,:], var1[0:self.nhalfswath,:]
                    , linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
            	ax1.pcolor(lon[self.nhalfswath::,:], lat[self.nhalfswath::,:], var1[self.nhalfswath::,:]
                    , linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
            	c1.set_clim([vmin, vmax])
		ax1.set_xlim([lomin, lomax])
		ax1.set_ylim([lamin, lamax])
                
		c2 = ax2.pcolor(lon[0:self.nhalfswath,:], lat[0:self.nhalfswath,:], var2[0:self.nhalfswath,:]
                    , linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
            	ax2.pcolor(lon[self.nhalfswath::,:], lat[self.nhalfswath::,:], var2[self.nhalfswath::,:]
                    , linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
            	c2.set_clim([vmin, vmax])
		ax2.set_xlim([lomin, lomax])
		ax2.set_ylim([lamin, lamax])

	    if land == 1:

            	my_map1 = Basemap(projection='merc', lat_0=(lamin+lamax)/2, lon_0=(lomin+lomax)/2, 
            resolution = 'l', ax = ax1,
            llcrnrlon = lomin, llcrnrlat= lamin,
            urcrnrlon = lomax, urcrnrlat = lamax, area_thresh = 10)
		
		my_map2 = Basemap(projection='merc', lat_0=(lamin+lamax)/2, lon_0=(lomin+lomax)/2
			, resolution = 'l', ax = ax2, llcrnrlon = lomin, llcrnrlat= lamin
			, urcrnrlon = lomax, urcrnrlat = lamax, area_thresh = 10)

            	x1, y1 = my_map1(lon, lat) # compute map proj coordinates
		x2, y2 = my_map2(lon, lat) # compute map proj coordinates
		
		c1 = my_map1.pcolor(x1[0:self.nhalfswath,:], y1[0:self.nhalfswath,:], var1[0:self.nhalfswath,:]
            		, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
		my_map1.pcolor(x1[self.nhalfswath::,:], y1[self.nhalfswath::,:], var1[self.nhalfswath::,:]
			, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
		c1.set_clim([vmin, vmax])

                c2 = my_map2.pcolor(x2[0:self.nhalfswath,:], y2[0:self.nhalfswath,:], var2[0:self.nhalfswath,:]
                	, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
                my_map2.pcolor(x2[self.nhalfswath::,:], y2[self.nhalfswath::,:], var2[self.nhalfswath::,:]
                	, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
                         
                c2.set_clim([vmin, vmax])
		
                if plotstyle == '1': 
                        my_map1.contour(x1[0:self.nhalfswath,:], y1[0:self.nhalfswath,:], var1[0:self.nhalfswath,:], levels, colors='w'
                         , linewidth=2)
			my_map1.contour(x1[self.nhalfswath::,:], y1[self.nhalfswath::,:], var1[self.nhalfswath::,:], levels, colors='w'
                         , linewidth=2)
                        my_map2.contour(x2[0:self.nhalfswath,:], y2[0:self.nhalfswath,:], var2[0:self.nhalfswath,:], levels, colors='w'                                       , linewidth=2)
                        my_map2.contour(x2[self.nhalfswath::,:], y2[self.nhalfswath::,:], var2[self.nhalfswath::,:], levels, colors='w'                                       , linewidth=2)		

		my_map1.drawcoastlines() 
            	my_map1.drawmapboundary()
            	my_map1.fillcontinents(color='#ddaa66')
            	my_map1.drawmeridians(np.arange(-160, 140, 2), labels=[1,0,0,1], size=18)
            	my_map1.drawparallels(np.arange(0, 70, 2), labels=[1,0,0,1], size=18)
            
            	my_map2.drawcoastlines() 
            	my_map2.drawmapboundary()
            	my_map2.fillcontinents(color='#ddaa66')
            	my_map2.drawmeridians(np.arange(-160, 140, 2), labels=[1,0,0,1], size=18)
            	my_map2.drawparallels(np.arange(0, 70, 2), labels=[1,0,0,1], size=18)
            
            cbar = plt.colorbar(c1, cax=axC, orientation='horizontal', extend=extend_opt)
            cbar.ax.tick_params(labelsize=16)

            return ax1, ax2

