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

import scipy.interpolate

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

def filter(var, iter_max, lambd, regularization_model):
    '''
    var = variable to be filtered
    iter_max = maximum number of iterations or 'cc' if convergence criteria is to be applied
    lambd = lambda: regularization parameter (the larger lambda, the more regular is the denoised image).  This parameter needs to be adapted to the regularization model. 
    regularization_model = 0 if Tikhonov regularization and 1 if second order penalization (regularization of vorticity)
    '''

    ima = var
    ima_obs = var

    if regularization_model == 0:
        tau = 1./(8*lambd)

        if iter_max == 'cc':
            epsilon = 0.000001
	    
            itern = 1
            imaf = ima + tau*(ima_obs - ima + lambd*np_laplacian(ima))

            numerator_norm = np.nansum(gradx(imaf - ima)**2) + np.nansum(grady(imaf - ima)**2)
            ima_norm = np.nansum(gradx(ima)**2) + np.nansum(grady(ima)**2)
            conv_crit = numerator_norm / ima_norm
        
            while conv_crit > epsilon:
                ima = np.copy(imaf)
                itern = itern + 1
                print 'iteration: ' + str(itera)
                imaf = ima + tau*(ima_obs - ima + lambd*np_laplacian(ima))
                numerator_norm = np.nansum(gradx(imaf - ima)**2) + np.nansum(grady(imaf - ima)**2)
                ima_norm = np.nansum(gradx(ima)**2) + np.nansum(grady(ima)**2)
                conv_crit = numerator_norm / ima_norm

        else:
            
            for k in xrange(0,iter_max):
                ima = ima + tau*(ima_obs - ima + lambd*laplacian(ima))
	    imaf = ima

    elif regularization_model == 1:
        
	tau=1./(512*lambd)
        
	if iter_max == 'cc':
            epsilon = 0.000001
	    
            itern = 1
            imaf = ima + tau*(ima_obs-ima+lambd*laplacian(laplacian(laplacian(ima))))

            numerator_norm = np.nansum(gradx(imaf - ima)**2) + np.nansum(grady(imaf - ima)**2)
            ima_norm = np.nansum(gradx(ima)**2) + np.nansum(grady(ima)**2)
            conv_crit = numerator_norm / ima_norm
        
            while conv_crit > epsilon:
                ima = np.copy(imaf)
                itern = itern + 1
                print 'iteration: ' + str(itera)
		imaf = ima + tau*(ima_obs - ima + lambd*laplacian(laplacian(laplacian(ima))))
                numerator_norm = np.nansum(gradx(imaf - ima)**2) + np.nansum(grady(imaf - ima)**2)
                ima_norm = np.nansum(gradx(ima)**2) + np.nansum(grady(ima)**2)
                conv_crit = numerator_norm / ima_norm

	else:

	    for k in xrange(0,iter_max):
                ima = ima + tau*(ima_obs-ima+lambd*laplacian(laplacian(laplacian(ima))))
            imaf = ima 

    return imaf

#=======================================================================

def plot_map(xx, yy, var, box, vmin, vmax, merv, parv, cmap, land, **options):
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
    - land = = 0 if you want only the data to be plotted or 1 if you want the land do be plotted ontop.
    - options:
      -- ax = axis (for e.g. if only one plot, not necessary)
    Output variables:
    - pp = plot object
    """
    lomin = box[0]
    lomax = box[1]
    lamin = box[2]
    lamax = box[3]
    
    if land =='0':

        if not options: #(i.e., if no axis define, one single plot)
                pp = plt.scatter(xx, yy, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
                pp.set_clim([vmin, vmax])
                cbar = plt.colorbar(pp)
                #cbar.ax.tick_params(labelsize=16)
        else:
                pp = ax.scatter(xx, yy, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, cmap = cmap)
                pp.set_clim([vmin, vmax])
                cbar = plt.colorbar(pp)
                #cbar.ax.tick_params(labelsize=16)

    elif land ==‘1’:

        if not options: #(i.e., if no axis define, one single plot) 
            map = Basemap(llcrnrlon=lomin,llcrnrlat=lamin,urcrnrlon=lomax,urcrnrlat=lamax,
            resolution='i', projection='merc', lat_0 = (lamin+lamax)/2, lon_0 = (lomin+lomax)/2)
	else:
            map = Basemap(llcrnrlon=lomin,llcrnrlat=lamin,urcrnrlon=lomax,urcrnrlat=lamax,
         resolution='i', projection='merc', lat_0 = (lamin+lamax)/2, lon_0 = (lomin+lomax)/2, ax=ax)

        map.drawcoastlines()
        map.fillcontinents(color='#ddaa66')
        map.drawmeridians(np.arange(-160, 140, 2), labels=merv, size=18)
        map.drawparallels(np.arange(0.5, 70.5, 2), labels=parv, size=18)

        pp = map.scatter(xx, yy, s=2, c=var, linewidth='0', vmin=vmin, vmax=vmax, latlon=True, cmap = cmap)
        pp.set_clim([vmin, vmax])
        if not options:
            cbar = plt.colorbar(pp)
            #cbar.ax.tick_params(labelsize=16)
    
    return pp

#=======================================================================

def plot_2_map(lon1, lat1, var1, lon2, lat2, var2, box, vmin, vmax, cmap):
    """
    Plots a 1x2 figure using scatter and basemap, and with a common colorbar showing 
    horizontally underneath the plots.
    Input variables:
    - lon1 = longitude of the left plot
    - lat1 = latitude of the left plot
    - var1 = variable of the left plot
    - lon2 = longitude of the right plot
    - lat2 = latitude of the right plot
    - var2 = variable of the right plot
    - box = box region of the area wanted to be shown
    , where box is a 1 x 4 array: 
    [minimum_longitude maximum_longitude minimum_latitude maximum_latitude]
    - vmin = minimum value of the colorbar 
    - vmax = maximum value of the colorbar
    - cmap = colormap to be used
    Output variables:
    - ax1 = left plot axis object
    - ax2 = right plot axis object
    You need to put the titles by using:
    ax1.set_title('title1', size=18)
    ax2.set_title('title2', size=18)
    """
    
    lomin = box[0]
    lomax = box[1]
    lamin = box[2]
    lamax = box[3]
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.95, .05], width_ratios=[1, 1])

    fig1 = plt.figure(figsize=(16, 10))  # figsize = (width ,height)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    axC = plt.subplot(gs[1, :])

    my_map1 = Basemap(projection='merc', lat_0=(lamin+lamax)/2, lon_0=(lomin+lomax)/2, 
    resolution = 'l', ax = ax1,
    llcrnrlon = lomin, llcrnrlat= lamin,
    urcrnrlon = lomax, urcrnrlat = lamax, area_thresh = 10)

    ny_t = var1.shape[0]; nx_t = var1.shape[1]
    lons_t, lats_t = my_map1.makegrid(nx_t, ny_t) # get lat/lons of ny by nx evenly space grid
    x, y = my_map1(lons_t, lats_t) # compute map proj coordinates

    my_map1.drawcoastlines() 
    my_map1.drawmapboundary()
    my_map1.fillcontinents(color='#ddaa66')
    my_map1.drawmeridians(np.arange(-160, 140, 2), labels=[1,0,0,1], size=18)
    my_map1.drawparallels(np.arange(0, 70, 2), labels=[1,0,0,1], size=18)

    c1 = my_map1.scatter(lon1, lat1, s=10, c=var1, latlon=True, linewidth='0', vmin=vmin
                         , vmax=vmax, cmap = cmap)
    c1.set_clim([vmin, vmax])

    my_map2 = Basemap(projection='merc', lat_0=(lamin+lamax)/2, lon_0=(lomin+lomax)/2
                      , resolution = 'l', ax = ax2, llcrnrlon = lomin, llcrnrlat= lamin
                      , urcrnrlon = lomax, urcrnrlat = lamax, area_thresh = 10)

    ny_t = var2.shape[0]; nx_t = var2.shape[1]
    lons_t, lats_t = my_map2.makegrid(nx_t, ny_t) # get lat/lons of ny by nx evenly space grid.
    x, y = my_map2(lons_t, lats_t) # compute map proj coordinates.

    my_map2.drawcoastlines() 
    my_map2.drawmapboundary()
    my_map2.fillcontinents(color='#ddaa66')
    my_map2.drawmeridians(np.arange(-160, 140, 2), labels=[1,0,0,1], size=18)
    my_map2.drawparallels(np.arange(0, 70, 2), labels=[1,0,0,1], size=18)

    c2 = my_map2.scatter(lon2, lat2, s=10, c=var2, latlon=True, linewidth='0', vmin=vmin
                         , vmax=vmax, cmap = cmap)
    c2.set_clim([vmin, vmax])

    cbar = plt.colorbar(c1, cax=axC, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    
    return ax1, ax2

#=======================================================================

def SWOT_filter(myfile, output_filename, lambd, iter_max, regularization_model, plot, savefig):
        
        output_file = Dataset(output_filename,'r+')
	
        ##====================== Load SWOT data ========================

	nc = Dataset(myfile)
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
	
        filt_SSH_model = np.concatenate((filt_SSH_model_left, filt_SSH_model_right),axis=1)
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

    		print 'Created the filtered variables'
	else:
    		print 'Variables already created'
	
	SSH_model_filtered = output_file.variables['SSH_model_filtered']
	SSH_model_filtered[:,:] = filt_SSH_model[:,:]
	SSH_obs_filtere = output_file.variables['SSH_obs_filtered']
	SSH_obs_filtere[:,:] = filt_SSH_obs[:,:]

	# Define long_name:
	if regularization_model == 0:
		SSH_model_filtered.reg_model = "SSH interpolated from model filtered using a first order penalization filter"
   		SSH_obs_filtered.reg_model = "SSH interpolated from model filtered using a first order penalization filter"
    
	elif regularization_model == 1:

    		SSH_model_filtered.long_name = "SSH interpolated from model filtered using a second order penalization filter"
	    	SSH_obs_filtered.long_name = "Observed SSH (SSH_model+errors) filtered using a second order penalization filter" 
        
	# Define filter parameter(s) as attribute:
        if iter_max == 'cc':
	
		SSH_model_filtered.Iter_max = 'Convergence criteria applied'
		SSH_obs_filtered.Iter_max =  'Convergence criteria applied'

	else:
		SSH_model_filtered.Iter_max = iter_max
		SSH_obs_filtered.Iter_max = iter_max

	SSH_model_filtered.Lambda = lambd
	SSH_obs_filtered.Lambda = lambd        
		
	print 'Variables updated'

	output_file.close()
        
        ##========================Plotting=========================
        if plot == 'yes':
		# Calculating the absolute difference between SSH_obs and _model:
		diff_nofilt = abs(SSH_obs - SSH_model)

		diff_filt = abs(filt_SSH_obs - filt_SSH_model)
		
                # Checking if pass 9 or 22:
		passnb = myfile.split('_p')[-1].split('.')[0] 
		
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


#=======================================================================

def tikhonov_filter(iter_max, var, mask, regularization_model, lambd):
    #initalize image to denoise from data
    ima_obs = np.copy(var)
    ima = np.copy(ima_obs)
    if mask==None:
        maskk=1
    else:
        maskk=mask
    if regularization_model == 0:
        tau = 1./(8*lambd)
        for k in xrange(0,iter_max):
            ima = ima+tau*(maskk*(ima_obs-ima)+lambd*laplacian(ima))
    elif regularization_model == 1:
        tau=1./(512*lambd)
        for k in xrange(0,iter_max):
            ima = ima + tau*(maskk*(ima_obs-ima)+lambd*laplacian(laplacian(laplacian(ima))))
    return ima

#=======================================================================

def SWOT_inpainting(myfile, output_filename, lambd, iter_max_filter, iter_max_inpainting, regularization_model, spline, plot, savefig):
        
     
        src = myfile
        dst = output_filename
        copyfile(src, dst)
	
	output_file = Dataset(output_filename,'r+')
	
        ##====================== Load SWOT data ========================

	nc = Dataset(myfile)
	lon = nc.variables['lon'][:]
	lat = nc.variables['lat'][:]
	lon_nadir = nc.variables['lon_nadir'][:]
    	lat_nadir = nc.variables['lat_nadir'][:]
	tim = nc.variables['time'][:]
	SSH_obs = nc.variables['SSH_obs'][:]
	SSH_model = nc.variables['SSH_model'][:]
	x_ac = nc.variables['x_ac'][:]
	x_al = nc.variables['x_al'][:]
	nc.close()
	
	
	##=================Preparing the SWOT data arrays===============
	# 1. Calculating the new size of variables
	pas = abs(x_ac[1]-x_ac[0]) #pas de x_ac
    	nhalfswath=np.shape(lon)[1]/2 #nombre de pixels dans une fauchee
    	dist=x_ac[nhalfswath]-x_ac[nhalfswath-1]-pas # distance en km entre les deux fauchees
    	newshape=int(x_ac.shape + dist/(pas))#new shape of x_ac
	ninter=dist/pas
	nac = np.shape(tim)[0]#new shape of time
	vx_ac=np.arange(x_ac[0],x_ac[-1]+pas,pas)
    	vtime=tim[:]
	vlon_nadir=lon_nadir[:]
    	vlat_nadir=lat_nadir[:]
	vx_al=x_al[:]

	# 2. Linear interpolation of latitude and longitude

	loninterp=scipy.interpolate.interp2d(np.insert(x_ac,nhalfswath,x_ac[0]+(x_ac[-1]-x_ac[0])/2), tim, np.insert(lon,nhalfswath,lon_nadir,axis=1), kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
    	vlon =np.insert(lon,nhalfswath,np.transpose(loninterp),axis=1)

    	latinterp=scipy.interpolate.interp2d(np.insert(x_ac,nhalfswath,x_ac[0]+(x_ac[-1]-x_ac[0])/2), tim, np.insert(lat,nhalfswath,lat_nadir,axis=1), kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
    	vlat =np.insert(lat,nhalfswath,np.transpose(latinterp),axis=1)

	if spline == 'yes' :
		# 3. Spline cubic interpolation of SSH after filter
	
			# 3.1 Masking the SWOT outputs
			SSH_model_ma = np.ma.masked_invalid(SSH_model)
			SSH_obs_ma = np.ma.masked_invalid(SSH_obs)


			# 3.2 Separate into left and right swath
			nhalfswath=np.shape(lon)[1]/2

			lon_left = lon[:,0:nhalfswath]
			lon_right = lon[:,nhalfswath:]
			lat_left = lat[:,0:nhalfswath]
			lat_right = lat[:,nhalfswath:]

			# 3.3 Masking the SWOT outputs and give a value of masked values
			SSH_model_left_ma = SSH_model_ma[:,0:nhalfswath]
			SSH_model_right_ma = SSH_model_ma[:,nhalfswath:]

			SSH_obs_left_ma = SSH_obs_ma[:,0:nhalfswath]
			SSH_obs_right_ma = SSH_obs_ma[:,nhalfswath:]
	
			SSH_model_left_0 = np.ma.filled(SSH_model_left_ma,0)
			SSH_model_right_0 = np.ma.filled(SSH_model_right_ma,0)
			SSH_obs_left_0 = np.ma.filled(SSH_obs_left_ma,0)
			SSH_obs_right_0 = np.ma.filled(SSH_obs_right_ma,0)
			
			
			# 3.4 Applying the filter
			filt_SSH_model_left = tikhonov_filter(iter_max=iter_max_filter, var=SSH_model_left_0,mask=None, regularization_model=regularization_model, lambd=lambd)
			filt_SSH_model_right = tikhonov_filter(iter_max=iter_max_filter, var=SSH_model_right_0,mask=None, regularization_model=regularization_model, lambd=lambd)
			filt_SSH_obs_left = tikhonov_filter(iter_max=iter_max_filter, var=SSH_obs_left_0, mask=None, regularization_model=regularization_model, lambd=lambd)
			filt_SSH_obs_right = tikhonov_filter(iter_max=iter_max_filter, var=SSH_obs_right_0, mask=None, regularization_model=regularization_model, lambd=lambd)


			# 3.5 Concatenating left and right swaths
        		filt_SSH_model = np.concatenate((filt_SSH_model_left, filt_SSH_model_right), axis=1)
			filt_SSH_obs = np.concatenate((filt_SSH_obs_left, filt_SSH_obs_right), axis=1)
			filt_SSH_model_ma = np.ma.masked_invalid(filt_SSH_model)
			filt_SSH_obs_ma = np.ma.masked_invalid(filt_SSH_obs)

			
			# 3.6 Spline cubic interpolation
			SSH_obs_interp=scipy.interpolate.interp2d(x_ac, tim, filt_SSH_obs_ma, kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
			vSSH_obs =np.ma.masked_values(np.insert(filt_SSH_obs_ma,nhalfswath,np.transpose(SSH_obs_interp),axis=1),SSH_obs.fill_value)

			SSH_model_interp=scipy.interpolate.interp2d(x_ac, tim, filt_SSH_model_ma, kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
			vSSH_model =np.ma.masked_values(np.insert(filt_SSH_model_ma,nhalfswath,np.transpose(SSH_model_interp),axis=1),SSH_model.fill_value)
	
	else :
	
			# 3.1 Add masked values in SSH_model and SSH_obs
	
    			vSSH_obs=np.ma.masked_values(np.insert(SSH_obs,nhalfswath,np.full((int((ninter)),int(nac)),SSH_obs.fill_value),axis=1),SSH_obs.fill_value) 
    			vSSH_model=np.ma.masked_values(np.insert(SSH_model,nhalfswath,np.full((int((ninter)),int(nac)),SSH_obs.fill_value),axis=1),SSH_obs.fill_value) 

	# 4. Masking the SWOT outputs and give a value of masked values
	SSH_model_mask = np.ma.masked_invalid(vSSH_model)
	SSH_obs_mask = np.ma.masked_invalid(vSSH_obs)
	SSH_model_0 = np.ma.filled(vSSH_model,0)
	SSH_obs_0 = np.ma.filled(vSSH_obs,0.)
	
	# 5. Creating the mask array
	mask_array=np.ones_like(vSSH_model)
	maskk=SSH_model_mask.mask
	mask_array[maskk]=0

	##=======================Applying the filter====================

	vfilt_SSH_obs = tikhonov_filter(iter_max = iter_max_inpainting, var= SSH_obs_0, mask=mask_array, regularization_model=regularization_model, lambd=lambd)
	vfilt_SSH_model = tikhonov_filter(iter_max = iter_max_inpainting, var= SSH_model_0, mask=mask_array, regularization_model=regularization_model, lambd=lambd)


	##===================Writing output NetCDF file=================
	#The output file has been created earlier (see above). Here we check whether the   variables we want to save already exist in this file, or not. If not, they are created in the NetCDF file. The filtered SSH fields are then saved in these variables.

	filename=output_filename + '.nc'
	filtset = Dataset(filename, 'w', format='NETCDF4')
	filtset.description = "fichier avec inpainting des valeurs de SSH"
	filtset.creator_name = "Audrey Monsimer"  

	# Creation des dimensions
	time=filtset.createDimension('time', nac)
	x_ac=filtset.createDimension('x_ac', newshape)

	#Creation des variables
	lat = filtset.createVariable('lat', 'f8', ('time','x_ac'))
	lat.long_name = "Latitude" 
	lat.units = "degrees_north"
	lat[:]=vlat
  
	lon = filtset.createVariable('lon', 'f8', ('time','x_ac'))
	lon.long_name = "longitude" 
	lon.units = "degrees_east"
	lon[:] = vlon

	lon_nadir = filtset.createVariable('lon_nadir', 'f8', ('time'))
	lon_nadir.long_name = "longitude nadir" 
	lon_nadir.units = "degrees_north"
	lon_nadir[:] = vlon_nadir

	lat_nadir = filtset.createVariable('lat_nadir', 'f8', ('time'))
	lat_nadir.long_name = "latitude nadir" 
	lat_nadir.units = "degrees_north"
	lat_nadir[:] = vlat_nadir

	time = filtset.createVariable('time', 'f8', ('time'))
	time.long_name = "time from beginning of simulation" 
	time.units = "days"
	time[:]= vtime

	x_al = filtset.createVariable('x_al', 'f8', ('time'))
	x_al.long_name = "Along track distance from the beginning of the pass" 
	x_al.units = "km"
	x_al[:] = vx_al

	x_ac = filtset.createVariable('x_ac', 'f8', ('x_ac'))
	x_ac.long_name = "Across track distance from nadir" 
	x_ac.units = "km"
	x_ac[:] = vx_ac

	SSH_obs = filtset.createVariable('SSH_obs', 'f8', ('time','x_ac'))
	SSH_obs.long_name = "Observed SSH (SSH_model+errors)" 
	SSH_obs.units = "m"
	SSH_obs[:] = SSH_obs

	SSH_model = filtset.createVariable('SSH_model', 'f8', ('time','x_ac'))
	SSH_model.long_name = "Observed SSH (SSH interpolated from model)" 
	SSH_model.units = "m"
	SSH_model[:] = SSH_model

	SSH_obs_filtered = filtset.createVariable('SSH_obs_filtered', 'f8', ('time','x_ac'))
	SSH_obs_filtered.long_name = "Observed SSH (SSH_model+errors) filtered" 
	SSH_obs_filtered.units = "m"
	SSH_obs_filtered[:] = vfilt_SSH_obs

	SSH_model_filtered = filtset.createVariable('SSH_model_filtered', 'f8', ('time','x_ac'))
	SSH_model_filtered.long_name = "Observed SSH (SSH interpolated from model) filtered" 
	SSH_model_filtered.units = "m"
	SSH_model_filtered[:] = vfilt_SSH_model

	filtset.close()  # close the new file

        
        ##========================Plotting=========================
        if plot == 'yes':
		box = [2, 7, 36, 44] #lomin, lomax, lamin, lamax

		plt.figure(figsize = (16, 10))

		ax1, ax2 = plot_2_map(vlon, vlat, vfilt_SSH_model, vlon, vlat,vfilt_SSH_obs, box, 			vmin=-0.3e0, vmax=0.3e0, cmap='jet')

		ax1.set_title('SSH_model', size=18)
		ax2.set_title('SSH_obs', size=18)
                
                if savefig == 'yes':
                    filenamep = output_filename.split('/')[-1].split('.')[0]
                    folder = output_filename.split('/Users')[-1].split(filenamep)[0]
                    savename = '/Users' + folder  + 'figs/' + filenamep + '.png'
                    plt.savefig(str(savename), bbox_inches='tight')
                
                plt.show()
        
        #return filt_SSH_model, filt_SSH_obs
