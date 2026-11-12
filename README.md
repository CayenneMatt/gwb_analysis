# A library for all your GWB analysis needs

This library requires that you have [holodeck](https://github.com/nanograv/holodeck) installed.

This code was used for the primary analysis and creation of figures in [Matt et al. (2026b)](https://ui.adsabs.harvard.edu/abs/2026ApJ..1006...63M/abstract) and [Matt et al. (2026c)](https://ui.adsabs.harvard.edu/abs/2026arXiv261028680M/abstract)

This code serves two primary functions:

1. Simplifying data handling for [holodeck](https://github.com/nanograv/holodeck) models and associated likelihoods
* Making corner plots
* Plotting GWB spectra
* Identifying best models and associated parameters

2. Turning SMBH populations associated with the GWB into quasar luminosity functions
* Taking SMBH mass functions and convolving with Eddington Ratio Distribution functions
* Modular design allows for easy testing of ERDFs
* Optional mass and redshift dependence for active fractions
