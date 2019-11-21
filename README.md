# SWOTmodule

* Example notebooks:

  * 2018-03-03-ec-discover-SWOTmodule.ipynb

  * 2018-04-18-lgn-discover-SWOTmodule.ipynb

  * 2018-04-18-lgn-discover-SWOTmodule_box_dataset.ipynb

  * 2018-05-21-ec-improve-varreg-convergence-merged.ipynb

* Codes:

  * SWOTdenoise.cfg: Configuration file called by SWOTdenoise.py.  Used to specify the name of the variables of the input file to be filtered.

  * SWOTdenoise.py: Module to read SWOT data, filter it and save it in a new output file (or not)
Working on making convergence faster, EC realized that in the former version of the variational method (SWOTdenoise_orig.py), the final solution was essentially the solution after Gaussian preconditioning for many sets of parameters (in particular including orders 2 and 3). So EC introduced a warm-start method.  See notebook: 2018-05-21-ec-improve-varreg-convergence.ipynb

* Example data:

  * sfs
  
  * sfs
