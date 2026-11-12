# A library for all your GWB analysis needs

This library requires that you have [holodeck](https://github.com/nanograv/holodeck) installed.

This code serves two primary functions:

1. Simplifying data handling for [holodeck](https://github.com/nanograv/holodeck) models and associated likelihoods
* Making corner plots
* Plotting GWB spectra
* Identifying best models and associated parameters

2. Turning SMBH populations associated with the GWB into quasar luminosity functions
* Taking SMBH mass functions and convolving with Eddington Ratio Distribution functions
* Modular design allows for easy testing of ERDFs
* Optional mass and redshift dependence for active fractions
