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
The two examples below are SWOT simulated passes from the NAtl60 model, generated for the fast-sampling phase in the western Mediterranean Sea.

  * MED_1km_nogap_JAS12_swotFastPhase_BOX_c01_p009_v2.nc: example SWOT dataset subregion (box_dataset) used in paper Gomez-Navarro et al., in prep.
  
  * MED_fastPhase_1km_swotFAST_c01_p009.nc: example SWOT dataset directly out of SWOT simulator (v xxx)
