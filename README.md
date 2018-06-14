# SWOTmodule

Working on making convergence faster, I realized that in the former version of the variational method, the final solution was essentially the solution after Gaussian preconditioning for many sets of parameters (in particular including orders 2 and 3). So I introduced a warm-start method.

Notebook: 2018-05-21-ec-improve-varreg-convergence

Unfortunately some functionalities appeared to be no more compatible with this new formulation, essentially in the corollary outputs norm and iter_max, because those are not unique anymore.
