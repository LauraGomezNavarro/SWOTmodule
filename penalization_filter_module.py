#!/usr/bin/env  python (to edit)
#=======================================================================
""" penalization_filter.py
Filters SWOT data using a first_order_penalization (Tikhonov regularization) filter for a given lambda and number of iterations
, and saves the filtered data in a netcdf file.
"""
#=======================================================================
import glob
from netCDF4 import Dataset
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from shutil import copyfile
import os

from scipy import ndimage as nd

import scipy.interpolate

#=======================================================================
sys.path.insert(0, "/Users/laura/PhD_private/develop/SWOT/")
import SWOT_data_class
#=======================================================================

def penalization_filter(var, iter_max='cc', lambd, regularization_model):
    '''
    var = variable to be filtered
    iter_max = maximum number of iterations or 'cc' if convergence criteria is to be applied.  (cc by default)
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
            conv_crit = np.nansum((imaf - ima)**2) / np.nansum(ima**2)

            while conv_crit > epsilon:
                ima = np.copy(imaf)
                itern = itern + 1
                print 'iteration: ' + str(itern)
                imaf = ima + tau*(ima_obs - ima + lambd*np_laplacian(ima)) 
                conv_crit = np.nansum((imaf - ima)**2) / np.nansum(ima**2)

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

            numerator_norm = np.nansum(gradi(imaf - ima)**2) + np.nansum(gradj(imaf - ima)**2)
            ima_norm = np.nansum(gradi(ima)**2) + np.nansum(gradj(ima)**2)
            conv_crit = numerator_norm / ima_norm
        
            while conv_crit > epsilon:
                ima = np.copy(imaf)
                itern = itern + 1
                print 'iteration: ' + str(itern)
                imaf = ima + tau*(ima_obs - ima + lambd*laplacian(laplacian(laplacian(ima))))
                numerator_norm = np.nansum(gradi(imaf - ima)**2) + np.nansum(gradj(imaf - ima)**2)
                ima_norm = np.nansum(gradi(ima)**2) + np.nansum(gradj(ima)**2)
                conv_crit = numerator_norm / ima_norm

        else:

            for k in xrange(0,iter_max):
                ima = ima + tau*(ima_obs-ima+lambd*laplacian(laplacian(laplacian(ima))))
            imaf = ima 

    return imaf

#=======================================================================

def SWOT_filter_box(_datadir, _namfiles_simu, _namdescr, output_filename, fill_gap='yes'): #, lambd, iter_max, regularization_model):
        """
        This function applies the penalization filter to SWOT data in which a subset (a box) is chosen.
        Input(s):
        - _datadir:
        - _namfiles_simu:
        - _namedescr:
        - output_filename:
        - fill_gap: .  It is 'yes' by default.
        Output(s):
        """
        #myfile instead of _datadir, _namfiles_simu, _namdescr, for alternative method without the class
        
        # Using the SWOT class:
        swot_pass = SWOT_data_class.swot_pass(_datadir, _namfiles_simu, _namdescr)
        swotfiles = swot_pass.get_list_of_swotfiles()
        
        swot_pass.read_data_ncfile_box(swotfiles[0]) #!
        
        """
        Alternative: without using SWOT class module:
        
        output_file = Dataset(output_filename,'r+')
        
        ##====================== Load SWOT data ========================

        nc = Dataset(myfile)
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        tim = nc.variables['time'][:]
        SSH_obs = nc.variables['SSH_obs'][:]
        SSH_model = nc.variables['SSH_model'][:]
        nc.close()
        """
        
        ##=================Preparing the SWOT data arrays===============
        # 1. Flipping first?
        # 1. Making SWOT array organized as (lon, lat) a.k.a., (x, y): 
        ## Not sure how to check this: --> for now ok, boxes saves as x, yso no need for this step
        """
        lon = np.transpose(lon)
        lat = np.transpose(lat)
        SSH_model = np.transpose(SSH_model)
        SSH_obs = np.transpose(SSH_obs)
        """

        # 2. Check longitude in -180 : +180 format:
        if any(xlo < 0 for xlo in swot_pass.lon):
            swot_pass.lon[swot_pass.lon > 180] -= 360 

        # 3. Flipping the SWOT array if necessary: 
        # So that lon increases from left to right and lat from bottom to top 
        # (depends on if pass is ascending or descending)

        lon_dif = swot_pass.lon[1,0] - swot_pass.lon[0,0]
        lat_dif = swot_pass.lat[0,1] - swot_pass.lat[0,0]

        # Need to revise below , above already adapted for datasets oragnized as x,y instead of y,x
        if (lat_dif<0):
            print 'flipped variables created'
            ## Ascending pass (pass 22 for in the fast-sampling phase)        
            lon = np.flipud(swot_pass.lon)
            lat = np.flipud(swot_pass.lat)
            SSH_model = np.flipud(swot_pass.SSH_model)
            SSH_obs = np.flipud(swot_pass.SSH_obs)

        elif (lon_dif<0):
            print 'flipped variables created'
            ## Descending pass (pass 9 for in the fast-sampling phase)
            lon = np.fliplr(swot_pass.lon)
            lat = np.fliplr(swot_pass.lat)
            SSH_model = np.fliplr(swot_pass.SSH_model)
            SSH_obs = np.fliplr(swot_pass.SSH_obs)

        # 4. Fixing the SWOT grid (regular longitude increments, i.e., no jump in swath gap or repeated nadir values)
        # At a 1km resolution, should have a size of 121 across track (size of first dim after transposing)    

        xx, yy = swot_pass.lon.shape
        if (swot_pass.lon[(xx/2)-1, 0] == swot_pass.lon [xx/2, 0]): # this only happens
        # when simulation is done with no gap
            print '_nogap variables created'
            # if no gap simulaton case need to eliminate repeated nadir value
            lon_nogap = np.delete(swot_pass.lon, (xx/2), axis=0)
            lat_nogap = np.delete(swot_pass.lat, (xx/2), axis=0)
            SSH_model_nogap = np.delete(swot_pass.SSH_model, (xx/2), axis=0)
            SSH_obs_nogap = np.delete(swot_pass.SSH_obs, (xx/2), axis=0)
            x_ac_nogap = np.delete(swot_pass.x_ac, (xx/2), axis=0)

        """
        check need of this below:
        # Obtain full land mask:
        SSH_model_nogap = np.ma.masked_invalid(SSH_model_nogap)    
        SSH_obs_nogap = np.ma.masked_invalid(SSH_obs_nogap)   
        mask_land = SSH_obs_nogap.mask

        # Convert gap values to 0:
        gap_indices = np.where((x_ac_nogap > -10) & (x_ac_nogap < 10))
        SSH_model_nogap[gap_indices[0]] = gap_values
        SSH_obs_nogap[gap_indices[0]] = gap_values
        """

        # 5. Masking the SWOT outputs
        ## 5.1. Change masked values to 0:
        mask_gap = swot_pass.SSH_obs.mask 

        swot_pass.SSH_model[mask_gap] = 0. 
        swot_pass.SSH_obs[mask_gap] = 0. 

        ## 5.2. Converting variables to masked array
        SSH_model_ma = np.ma.masked_array(swot_pass.SSH_model, mask_gap)
        SSH_obs_ma = np.ma.masked_array(swot_pass.SSH_obs, mask_gap)

        ## 5.3. Applying the gaussian filter:

        sigma = 10 # --> this sigma?
        SSH_model_gausf = gaussian_with_nans(SSH_model_ma, sigma)
        SSH_obs_gausf = gaussian_with_nans(SSH_obs_ma, sigma)

        ## 5.4 Assigning the gaussian filtered data to the gap:

        SSH_model_gap_full = SSH_model_ma.copy()
        SSH_obs_gap_full = SSH_obs_ma.copy()

        SSH_model_gap_full[mask_gap] = SSH_model_gausf[mask_gap]
        SSH_obs_gap_full[mask_gap] = SSH_obs_gausf[mask_gap] # or leave it as SSH_obs_nogap.mask? and one variable less?

        # No need of this step here for this dataset, will implement this check later:
        """
        # 5. Creating the gap values array if necessary:
        # The half-swath gap values will be set to 0 and masked.

        SWOT_grid_resolution = abs(x_ac[1] - x_ac[0]) # pixel size in km
        gap_size = 20. # size of the the half-swath gap in km
        gap_pixels = (gap_size / SWOT_grid_resolution) - 1 # -1 so that no repeat nadir value
        gap_values = np.ma.zeros((gap_pixels, yy))
        gap_values.mask = True
        gap_values.fill_value = 0
        """
        
        ##=======================Applying the filter====================

        filt_SSH_model  = penalization_filter(SSH_model_gap_full, iter_max, lambd, regularization_model)
        filt_SSH_obs    = penalization_filter(SSH_obs_gap_full, iter_max, lambd, regularization_model)
        
        ## Mask gap or not (i.e. show inpainting or not):
        if fillgap == 'yes':
            print 'Inpainting applied'
        elif fillgap == 'no':
            filt_SSH_model = np.ma.masked_array(filt_SSH_model, mask_gap)
            filt_SSH_obs = np.ma.masked_array(filt_SSH_obs, mask_gap)
        
        """
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
        SSH_obs_filtered = output_file.variables['SSH_obs_filtered']
        SSH_obs_filtered[:,:] = filt_SSH_obs[:,:]

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
        """
#=======================================================================

def tikhonov_filter(iter_max, var, mask, regularization_model, lambd):
    #initalize image to denoise from data
    ima_obs = np.copy(var)
    ima = np.copy(ima_obs)
    if mask == None:
        maskk = 1
    else:
        maskk = mask
    if regularization_model == 0:
        tau = 1./(8*lambd)
        for k in xrange(0,iter_max):
            ima = ima+tau*(maskk*(ima_obs-ima)+lambd*laplacian(ima))
    elif regularization_model == 1:
        tau = 1./(512*lambd)
        for k in xrange(0, iter_max):
            ima = ima + tau*(maskk*(ima_obs-ima) + lambd*laplacian(laplacian(laplacian(ima))))
    return ima


#=======================================================================

def moyenne(var,r,i,j):
    
    ''' Cette fonction permet de calculer la moyenne des points dans un rayon de r autour du point (i,j).
    Lorsque que l on veut calculer la moyenne autour d un point situe sur le bord de la trace, 
    ce n'est plus le rayon r qui est utilise mais les rayons rimd, rimg, rjmd et rjmg. '''
    
    newvar=np.ma.copy(var)
    rimd=min(r,newvar.shape[0]-i-1) #rayon sur la ligne a droite
    rimg=min(r,i)                   #rayon sur la ligne a gauche
    rjmd=min(r,newvar.shape[1]-j-1) #rayon sur la colonne en haut
    rjmg=min(r,j)                   #rayon sur la colonne en bas
    moy=np.ma.mean(newvar[i-rimg:i+rimd+1,j-rjmg:j+rjmd+1])
    return moy

#=======================================================================

def demaskk(variable):
    
    ''' 
    Cette fonction permet d appliquer la fonction moyenne en chaque point masque de la trace
    '''
    
    r=1
    var=np.ma.copy(variable)
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            if variable.mask[i,j] == True:  #on ne prend en compte que les points masques
                r=r-1  #ceci permet de repartir du r precedent plutot que de r=1
                while moyenne(variable,r,i,j) is np.ma.masked:
                    r=r+1
                var[i,j]=moyenne(variable,r,i,j)
    return var

#=======================================================================

def SWOT_inpainting(myfile, output_filename, lambd=40., iter_max_filter=100, iter_max_inpainting=100, regularization_model=0, demask='no', spline='no'):
        
        ''' 
        myfile : fichier a traiter
        output_filename : nom du fichier NetCDF du resultat de cette fonction
        lambd (optionnel) : valeur de lambda pour la fonction cout, par defaut lambd=40.
        iter_max_filter (optionnel) : nombre d'iterations pour le filtre de debruitage, par defaut iter_max_filter=100
        iter_max_inpainting (optionnel) : nombre d'iterations pour le filtre d'inpainting,  par defaut iter_max_inpainting=100
        regularization_model(optionnel): valeur de regularization_model pour le filtre, par defaut regularization_model=0
        demask='yes' or 'no'(optionnel): si demask='yes' les valeurs masquees des iles ou des continents seront remplacees par la valeur reelle la plus proche, par defaut demask='no'
        spline='yes' or 'no'(optionnel): si spline='yes' l'inter-trace sera remplie par une interpolation spline cubique, par defaut spline='no'
        '''
        
        src = myfile
        dst = output_filename
        copyfile(src, dst)

        output_file = Dataset(output_filename,'r+')

        if demask == 'yes':
            print ('Attention, cela peut prendre du temps.')

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
        pas = abs(x_ac[1] - x_ac[0]) #pas de x_ac
        nhalfswath = np.shape(lon)[1] /2 #nombre de pixels dans une fauchee
        dist = x_ac[nhalfswath] - x_ac[nhalfswath-1]-pas # distance en km entre les deux fauchees
        newshape = int(x_ac.shape + dist/(pas))#new shape of x_ac
        ninter = dist / pas
        nac = np.shape(tim)[0]#new shape of time
        vx_ac = np.arange(x_ac[0],x_ac[-1]+pas,pas)
        vtime = tim[:]
        vlon_nadir = lon_nadir[:]
        vlat_nadir = lat_nadir[:]
        vx_al = x_al[:]

        # 2. Linear interpolation of latitude and longitude

        loninterp = scipy.interpolate.interp2d(np.insert(x_ac,nhalfswath,x_ac[0]+(x_ac[-1]-x_ac[0])/2), tim, np.insert(lon,nhalfswath,lon_nadir,axis=1), kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
        vlon = np.insert(lon,nhalfswath,np.transpose(loninterp),axis=1)

        latinterp = scipy.interpolate.interp2d(np.insert(x_ac,nhalfswath,x_ac[0]+(x_ac[-1]-x_ac[0])/2), tim, np.insert(lat,nhalfswath,lat_nadir,axis=1), kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas,x_ac[nhalfswath],pas),tim)
        vlat = np.insert(lat,nhalfswath,np.transpose(latinterp),axis=1)

        if spline == 'yes' :
        # 3. Spline cubic interpolation of SSH after filter
        
                # 3.1 Masking the SWOT outputs
                SSH_model_ma = np.ma.masked_invalid(SSH_model)
                SSH_obs_ma = np.ma.masked_invalid(SSH_obs)

                if demask == 'yes':
                    # 3.4 Demask
                    new_SSH_model = demaskk(SSH_model_ma)
                    new_SSH_obs=demaskk(SSH_obs_ma)
                else :
                    new_SSH_model = SSH_model_ma
                    new_SSH_obs=SSH_obs_ma

                # 3.2 Separate into left and right swath
                nhalfswath=np.shape(lon)[1]/2

                lon_left = lon[:,0:nhalfswath]
                lon_right = lon[:,nhalfswath:]
                lat_left = lat[:,0:nhalfswath]
                lat_right = lat[:,nhalfswath:]

                # 3.3 Masking the SWOT outputs and give a value of masked values
                SSH_model_left_ma = new_SSH_model[:,0:nhalfswath]
                SSH_model_right_ma = new_SSH_model[:,nhalfswath:]

                SSH_obs_left_ma = new_SSH_obs[:,0:nhalfswath]
                SSH_obs_right_ma = new_SSH_obs[:,nhalfswath:]

                if demask == 'yes':
                    new_SSH_model_left = SSH_model_left_ma
                    new_SSH_model_right = SSH_model_right_ma
                    new_SSH_obs_left=SSH_obs_left_ma
                    new_SSH_obs_right=SSH_obs_right_ma
                else:
                    new_SSH_model_left = np.ma.filled(SSH_model_left_ma,0)
                    new_SSH_model_right = np.ma.filled(SSH_model_right_ma,0)
                    new_SSH_obs_left=np.ma.filled(SSH_obs_left_ma,0)
                    new_SSH_obs_right=np.ma.filled(SSH_obs_right_ma,0)

                # 3.4 Applying the filter

                filt_SSH_model_left = tikhonov_filter(iter_max=iter_max_filter, var=new_SSH_model_left,mask=None, regularization_model=regularization_model, lambd=lambd)
                filt_SSH_model_right = tikhonov_filter(iter_max=iter_max_filter, var=new_SSH_model_right,mask=None, regularization_model=regularization_model, lambd=lambd)
                filt_SSH_obs_left = tikhonov_filter(iter_max=iter_max_filter, var=new_SSH_obs_left, mask=None, regularization_model=regularization_model, lambd=lambd)
                filt_SSH_obs_right = tikhonov_filter(iter_max=iter_max_filter, var=new_SSH_obs_right, mask=None, regularization_model=regularization_model, lambd=lambd)

                filt_SSH_model_left_ma = np.ma.masked_invalid(filt_SSH_model_left)
                filt_SSH_model_right_ma = np.ma.masked_invalid(filt_SSH_model_right)
                filt_SSH_obs_left_ma = np.ma.masked_invalid(filt_SSH_obs_left)
                filt_SSH_obs_right_ma = np.ma.masked_invalid(filt_SSH_obs_right)

                # 3.5 Concatenating left and right swaths
                filt_SSH_model = np.concatenate((new_SSH_model_left, new_SSH_model_right), axis=1)
                filt_SSH_obs = np.concatenate((new_SSH_obs_left, new_SSH_obs_right), axis=1)
                filt_SSH_model_ma = np.ma.masked_invalid(filt_SSH_model)
                filt_SSH_obs_ma = np.ma.masked_invalid(filt_SSH_obs)

                # 3.6 Spline cubic interpolation
                SSH_obs_interp = scipy.interpolate.interp2d(x_ac, tim, filt_SSH_obs_ma, kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas, x_ac[nhalfswath], pas), tim)
                vSSH_obs = np.ma.masked_values(np.insert(filt_SSH_obs_ma, nhalfswath, np.transpose(SSH_obs_interp), axis=1), SSH_obs.fill_value)

                SSH_model_interp = scipy.interpolate.interp2d(x_ac, tim, filt_SSH_model_ma, kind='cubic', copy=True, bounds_error=False)(np.arange(x_ac[nhalfswath-1]+pas, x_ac[nhalfswath], pas), tim)
                vSSH_model = np.ma.masked_values(np.insert(filt_SSH_model_ma, nhalfswath, np.transpose(SSH_model_interp), axis=1), SSH_model.fill_value)

        else :

                # 3.1 Add masked values in SSH_model and SSH_obs

                vSSH_obs = np.ma.masked_values(np.insert(SSH_obs, nhalfswath,np.full((int((ninter)), int(nac)), SSH_obs.fill_value), axis=1), SSH_obs.fill_value) 
                vSSH_model = np.ma.masked_values(np.insert(SSH_model, nhalfswath, np.full((int((ninter)), int(nac)),SSH_obs.fill_value), axis=1), SSH_obs.fill_value) 

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

        vfilt_SSH_model = tikhonov_filter(iter_max = iter_max_inpainting, var= SSH_model_0, mask=mask_array, regularization_model=regularization_model, lambd=lambd)
        vfilt_SSH_obs = tikhonov_filter(iter_max = iter_max_inpainting, var= SSH_obs_0, mask=mask_array, regularization_model=regularization_model, lambd=lambd)
        
        return vfilt_SSH_model, vfilt_SSH_obs
    
        """
        ##===================Writing output NetCDF file=================
        # The output file has been created earlier (see above). Here we check whether the   variables we want to save already exist in this file, or not. If not, they are created in the NetCDF file. The filtered SSH fields are then saved in these variables.

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
        """
        
