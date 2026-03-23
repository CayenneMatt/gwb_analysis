# Class to aid in GWB model analysis
import numpy as np
import scipy.stats as stats

import holodeck as holo
import holodeck.sams as sams
from holodeck.constants import MSOL
from holodeck import librarian

import kalepy as kale
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from numpy import trapz
import astropy.units as u
import astropy.constants as c

import copy

# import pytensor.tensor as pt

"""
TODO:
* Figure out how to get this to interface with PyMC
"""

class Model_Info(object):
    """
    Class to analyze gravitational wave background models generated using holodeck.
    This class automatically reads the data and assigns attributes for the GWB amplitudes, parameters, likelihoods, and frequencies.

    Parameters
    -----------
    path : str
        path to the data, should end with a '/' and be the same as the path used to generate the data
    file : str
        filename of the data, should be the same as the filename used to generate the data
    model_name:  str
        name of the model, used for plotting
    color : str
        color for all plotting associated with this model
    line_style : str
        line style for all plotting associated with this model

    threshold : float, optional
        minimum likelihood cutoff, used for plotting error bars on the spectrum. Default is 0.5
    evolving : bool, optional
        whether the model has MMBulge evolution, used for plotting. Default is False
    stdev : float, optional
        standard deviation of evolution parameter, used for plotting, should be None if evolving is False. Default is None
    nfreq : int, optional
        number of frequency bins to use in likelihood calculation, 5 is recommended. Default is 5
    param_space_name : str, optional
        name of parameter space, should be the same as the parameter space used to generate the data. Default is 'PS_Astro_Strong_All'

    Attributes
    -----------
    path : str
        path to the data specified as an argument
    file : str
        filename of the data specified as an argument
    param_space_name : str
        name of parameter space specified as an argument
    gwb : array-like
        GWB amplitudes for each model, shape is (nsamples, nrealizations, nfreqs)
    param_names : array-like
        names of parameters in the model
    params : array-like
        parameter sample associated with each model
    ln_like : array-like
        log-likelihood of each model
    freqs : array-like
        frequencies at which the amplitudes are calculated
    model_name : str
        model name specified as an argument
    color :  str
        color for all plotting associated with this model specified as an argument
    line_style : str
        line style for all plotting associated with this model specified as an argument
    threshold : float
        minimum likelihood cutoff specified as an argument
    evolving : bool
        whether the model has MMBulge evolution specified as an argument
    stdev : float
        standard deviation of evolution parameter specified as an argument
    nfreq : int
        number of frequency bins to use in likelihood calculation specified as an argument
    idcs : array-like
        index of evolution parameter, None if evolving is False
    space_class : class
        class of the parameter space, used to get priors and fiducial values
    fiducial_values:  dict
        dictionary of fiducial values for each parameter in the model
    posteriors : dict
        dictionary of median posterior values for each parameter in the model,  (!!!) calculated by get_posteriors(),
        will be None until get_posteriors() is called
    posteriors_err : dict
        dictionary of standard deviation of posterior values for each parameter in the model,  (!!!) calculated by get_posteriors(),
        will be the same as hardcoded default errors until get_posteriors() is called
    plt_labels : dict
        dictionary of plot labels associated with each parameter, used for plotting
    
    Methods
    --------
    get_priors() :
        returns parameter names and default values from the prior parameter space.
    get_posteriors() :
        get the median of the posterior distributions for each parameter in the model and store them in the posteriors attribute.
    corner_plot() :
        Old and slow, but tested version of corner plot, will be removed in favor of corner_plot_fast() pending appropriate testing.
    corner_plot_fast() :
        faster version of corner plot, should produce the same plot as corner_plot() but much faster, recommended over corner_plot() pending appropriate testing.
    add_spectrum() :
        add the spectrum of the model to a given axis, with optional error bars and label.
    bhmf() :
        produce the black hole mass function at a given redshift, calculated using holodeck by convolving a double Schechter GSMF with an Mbh–M_bulge relation,
        using either the posterior values or the fiducial values for the parameters in the model.
    bhmf_err() :
        produce the error on the black hole mass function at
        a given redshift, calculated using holodeck by convolving a double Schechter GSMF with an Mbh–M_bulge relation, using either the posterior values or the
        fiducial values for the parameters in the model, and using the standard deviation of the posterior values as the error on each parameter to calculate the error on the BHMF.
    gsmf() :
        produce the galaxy stellar mass function at a given redshift, calculated using holodeck by convolving a double Schechter GSMF with an Mbh–M_bulge relation,
        using either the posterior values or the fiducial values for the parameters in the model.
    
    get_shenf() :
        retrieve the Shen et al. (2020) QLF fit at a given redshift.
    get_shend() :
        retrieve the Shen et al. (2020) QLF data at a given redshift.
    fdfunc() :
        calculate the AGN fraction as a function of black hole mass and redshift, fit to data in Zou et al. (2024)
    bhmf_from_gsmf() :
        calculate the black hole mass function from the galaxy stellar mass function using the MMBulge relation
    calculate_radiative_efficiency() :
        calculate the radiative efficiency as a function of black hole mass at a given redshift.
    bhargal() :
        calculate black hole accretion rate as a function of black hole mass and redshift.
    eta_from_mbh_davis() :
        calculate radiative efficiency using the relation from Davis & Laor (2011) as a function of black hole mass.
    eta_from_mbh_line() :
        calculate radiative efficiency as a funciton of black hole mass using a linear fit to Li et al. (2012) data.
    eta_from_mbh_logistic() :
        calculate radiative efficiency as a function of black hole mass using a logistic fit to Li et al. (2012) data.
    L_from_Mbh_via_mdot_eta_func() :
        calculate AGN luminosity from black hole mass and accretion rate, using a specified function for the radiative efficiency.

    """
    # Attributes
    def __init__(self, path, file, model_name, color, line_style, threshold=0.5, evolving=False, stdev=None, nfreq=5, param_space_name = 'PS_Astro_Strong_All'):

        self.path = path  # Path to data
        self.file = file  # Filename
        self.param_space_name = param_space_name  # Name of parameter space

        # Data
        dat = np.load(self.path+self.file)
        self.gwb = dat['gwb']  # GWB amplitudes for each model
        self.param_names = dat['names']  # Names of parameters in the model
        self.params = dat['params']  # Parameter sample associated with each model
        self.ln_like = dat['ln_like']  # Log-likelihood of each model
        self.freqs = np.array(dat['fobs_cents'])  # Frequencies for all models

        # Plotting and book keeping
        self.model_name = model_name  # Model name
        self.color = color  # Color for all plotting associated with this model
        self.line_style = line_style  # Line style for all plotting associated with this model
        self.threshold = threshold  # Likelihood value above which are ~1% of all models
        self.evolving = evolving  # Whether the model has MMBulge evolution
        self.stdev = stdev  # Standard deviation of evolution parameter
        self.nfreq = nfreq  # Number of frequency bits to fit to. Default is 5
        self.idcs = None  # Index of evolution parameter

        id = np.where((self.param_names == 'mmb_zplaw') | (self.param_names == 'mmb_zplaw_amp') | (self.param_names == 'mmb_zplaw_slope') | (self.param_names == 'mmb_zplaw_scatter'))[0]
        if len(id) > 0:
            self.idcs = id
        
        # Dictionary of priors to be modified by get_posteriors(), note that LM* models have different priors, but currently sample every parameter.
        self.space_class = librarian.param_spaces_dict[self.param_space_name]
        self.fiducial_values = copy.deepcopy(self.space_class.DEFAULTS)
        self.posteriors = copy.deepcopy(self.space_class.DEFAULTS)
        self.posteriors = {k: None for k in self.posteriors.keys()}

        self.posteriors_err = {'hard_time': 3,
                               'hard_sepa_init': 5,
                                'hard_rchar': 3.0,
                                'hard_gamma_outer': 0.5,
                                'hard_gamma_inner': 0.5,
                                'gsmf_log10_phi_one_z0': 0.028,
                                'gsmf_log10_phi_one_z1': 0.072,
                                'gsmf_log10_phi_one_z2': 0.031,
                                'gsmf_log10_phi_two_z0': 0.050,
                                'gsmf_log10_phi_two_z1': 0.070,
                                'gsmf_log10_phi_two_z2': 0.020,
                                'gsmf_log10_mstar_z0': 0.026,
                                'gsmf_log10_mstar_z1': 0.045,
                                'gsmf_log10_mstar_z2': 0.015,
                                'gsmf_alpha_one': 0.070,
                                'gsmf_alpha_two': 0.150,
                                'gmr_norm0_log10': 0.0045,
                                'gmr_normz': 0.0128,
                                'gmr_malpha0': 0.00338,
                                'gmr_malphaz': 0.0316,
                                'gmr_mdelta0': 0.0202,
                                'gmr_mdeltaz': 0.0440,
                                'gmr_qgamma0': 0.0026,
                                'gmr_qgammaz': 0.0021,
                                'gmr_qgammam': 0.0013,
                                'mmb_mamp_log10': 0.05,
                                'mmb_plaw': 0.08,
                                'mmb_scatter_dex': 0.05,
                                'mmb_zplaw_amp': 0.0,
                                'mmb_zplaw_slope': 0.0,
                                'mmb_zplaw_scatter': 0.0,
                                'bf_frac_lo': 0.4,
                                'bf_frac_hi': 1.0,
                                'bf_mstar_crit': 0.5,
                                'bf_width_dex': 0.1}

        # Dictionary of plot labels associated with each parameter
        self.plt_labels = {'hard_time'           : r"$\tau_\mathrm{f}$",
                            'hard_sepa_init'        : r"Bin. Sep.",
                            'hard_rchar'            : r'R$_{char}$',
                            'hard_gamma_inner'      : r"$\nu_{inner}$",
                            'hard_gamma_outer'      : r"$\nu_{outer}$",
                            'gsmf_phi0_log10'       : r"$\log \phi_{*}$",
                            'gsmf_mchar0_log10'     : r"$\log M_{\mathrm{c}}$",
                            'gsmf_log10_phi_one_z0' : r"$\log \phi_{*, 1,0}$",
                            'gsmf_log10_phi_one_z1' : r"$\phi_{*, 1,1}$",
                            'gsmf_log10_phi_one_z2' : r"$\phi_{*, 1,2}$",
                            'gsmf_log10_phi_two_z0' : r"$\log \phi_{*, 2,0}$",
                            'gsmf_log10_phi_two_z1' : r"$\phi_{*, 2,1}$",
                            'gsmf_log10_phi_two_z2' : r"$\phi_{*, 2,2}$",
                            'gsmf_log10_mstar_z0'   : r"$M_{\mathrm{c} ,0}$",
                            'gsmf_log10_mstar_z1'   : r"$M_{\mathrm{c} ,1}$",
                            'gsmf_log10_mstar_z2'   : r"$M_{\mathrm{c} ,2}$",
                            'gsmf_alpha_one'        : r"$\alpha_1$",
                            'gsmf_alpha_two'        : r"$\alpha_2$",
                            'gmr_norm0_log10'       : r'$\log A_{GMR, 0}$', # r'gmr_norm0_log10',
                            'gmr_normz'             : r'$\eta_{GMR}$', # r'gmr_normz',
                            'gmr_malpha0'           : r'$\alpha_{GMR, 0}$', # r'gmr_malpha0',
                            'gmr_malphaz'           : r'$\alpha_{GMR, 1}$', # r'gmr_malphaz',
                            'gmr_mdelta0'           : r'$\delta_{GMR, 0}$', # r'gmr_mdelta0',
                            'gmr_mdeltaz'           : r'$\delta_{GMR, 1}$', # r'gmr_mdeltaz',
                            'gmr_qgamma0'           : r'$\beta_{GMR, 0}$', # r'gmr_qgamma0',
                            'gmr_qgammaz'           : r'$\beta_{GMR, 1}$', # r'gmr_qgammaz',
                            'gmr_qgammam'           : r'$\gamma_{GMR}$', # r'gmr_qgammam',
                            'mmb_mamp_log10'        : r"$\alpha_{0}$",
                            'mmb_plaw'              : r"$\beta_{0}$",
                            'mmb_zplaw'             : r'$\alpha_z$',
                            'mmb_zplaw_amp'         : r'$\alpha_z$',
                            'mmb_zplaw_slope'       : r'$\beta_z$',
                            'mmb_zplaw_scatter'     : r'$\epsilon_z$',
                            'mmb_scatter_dex'       : r'$\epsilon_{0}$',
                            'bf_frac_lo'            : r'$\mathrm{f}_\mathrm{bulge, lo}$', # bf_frac_lo
                            'bf_frac_hi'            : r'$\mathrm{f}_\mathrm{bulge, hi}$', # bf_frac_hi
                            'bf_mstar_crit'         : r'$\mathcal{M}_{* , crit}$', # bf_mstar_crit
                            'bf_width_dex'          : r'$W_\mathrm{bf}$'} # bf_width_dex
        
    def get_priors(self):
        """
        returns parameter names and sampled distributions from the prior parameter space
        
        Sampled distribution of parameter i: space.param_samples.transpose()[i]
        Name of parameter i: space.param_names[i]
        """

        # space_class = librarian.param_spaces_dict[self.param_space_name]
        space = self.space_class(nsamples=int(1e4)) # Draw samples from the parameter space with LHC

        return space.param_names, space.param_samples.transpose()

    def get_posteriors(self):
        """
        Get the median posterior values for each parameter in the model
        """
        print('Calulating posteriors for model: {}'.format(self.model_name))
        skip = None

        gwb_med = np.median(self.gwb, axis=-1)
        valid = np.any(gwb_med > 0, axis=1)

        like_med = np.sum(self.ln_like[::skip, :self.nfreq], axis=1)
        weights_med = np.exp(like_med[valid])

        for jj, n in enumerate(self.param_names):
            xx = self.params[:, jj]
            xx = xx[valid]
            rv = stats.rv_histogram(np.histogram(xx, weights=weights_med))
            median = rv.median()
            stdev = rv.std()
            self.posteriors[n] = median
            self.posteriors_err[n] = stdev

        return self
    
    def corner_plot(self, nbins=20, cmap='Blues'):
        """
        Old and slow, but tested version of corner plot, will be removed in favor of corner_plot_fast() pending appropriate testing.
        """
        print('Depricated, use corner_plot_fast() instead.')

        npars = len(self.param_names)
        nsamp, npars = self.params.shape

        gwb_med = np.median(self.gwb, axis=-1)

        valid = np.any(gwb_med > 0, axis=1)
        like = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sort_idx = np.argsort(like)
        ww = np.exp(like[sort_idx[valid]])

        colorsc = mpl.colormaps[cmap](np.linspace(0.02, 0.98, nsamp))

        nspec = 100
        lnl_extr = np.percentile(like[sort_idx[-nspec:]], [5, 100])
        _alpha_map_1 = lambda xx: (xx - lnl_extr[0])/(lnl_extr[1] - lnl_extr[0])
        _alpha_map_2 = lambda xx: np.clip(_alpha_map_1(xx), 0.02, 1.0)
        _alpha_map_3 = lambda xx: np.power(_alpha_map_2(xx), 1)
        alpha_map = lambda xx: 0.5*_alpha_map_3(xx)
        alphas = alpha_map(like[sort_idx])
        colorsc[:, -1] = alphas

        figsize = 2*npars
        fig, axes = plt.subplots(figsize=[figsize, figsize], ncols=npars, nrows=npars, sharex='col')  # , sharey='row')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        edges = []
        extrema = []
        for ii in range(npars):
            xx = self.params[:, ii][valid]
            extr = holo.utils.minmax(xx)
            extrema.append(extr)
            ee = np.linspace(*extr, nbins)
            edges.append(ee)
            axes[ii, ii].set(xlim=extr)
            axes[-1, ii].set(xlabel=self.plt_labels[self.param_names[ii]])
            if ii > 0:
                axes[ii, 0].set(ylabel=self.plt_labels[self.param_names[ii]])
            for jj in range(ii):
                ax = axes[ii, jj]
                ax.set(ylim=extr)

        for (ii, jj), ax in np.ndenumerate(axes):
            if jj > ii:
                ax.set_visible(False)
                continue
            ax.grid(True, alpha=0.25)
            xx = self.params[:, jj][sort_idx[valid]]

            if jj > 0 and ii != jj:
                ax.set_yticklabels(['' for ii in ax.get_yticks()])

            if ii == jj:
                ax.hist(xx, histtype='step', bins=edges[jj], alpha=0.5, density=True, color='k', ls='--')
                ax.hist(xx, histtype='step', bins=edges[jj], weights=ww, alpha=0.5, density=True, color=colorsc[-1], lw=1.5)

                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                continue

            yy = self.params[:, ii][sort_idx[valid]]
            bins = (edges[jj], edges[ii])
            kale.contour((xx, yy), edges=bins, weights=ww, ax=ax, pad=0, smooth=1.5);


    def corner_plot_fast(self, nbins=20, cmap='Blues'):
        """
        Faster version of corner plot
        """

        npars = len(self.param_names)
        nsamp, npars = self.params.shape

        gwb_med = np.median(self.gwb, axis=-1)

        valid = np.any(gwb_med > 0, axis=1)
        like = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sort_idx = np.argsort(like)
        ww = np.exp(like[sort_idx[valid]])

        colorsc = mpl.colormaps[cmap](np.linspace(0.02, 0.98, nsamp))

        nspec = 100
        lnl_extr = np.percentile(like[sort_idx[-nspec:]], [5, 100])
        _alpha_map_1 = lambda xx: (xx - lnl_extr[0])/(lnl_extr[1] - lnl_extr[0])
        _alpha_map_2 = lambda xx: np.clip(_alpha_map_1(xx), 0.02, 1.0)
        _alpha_map_3 = lambda xx: np.power(_alpha_map_2(xx), 1)
        alpha_map = lambda xx: 0.5*_alpha_map_3(xx)
        alphas = alpha_map(like[sort_idx])
        colorsc[:, -1] = alphas

        figsize = 2*npars

        fig = plt.figure(figsize=(figsize, figsize))
        axes = {}

        for i in range(npars):
            for j in range(i + 1):
                ax = fig.add_subplot(npars, npars, i*npars + j + 1)
                axes[(i, j)] = ax
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        edges = []
        extrema = []
        for ii in range(npars):
            xx = self.params[:, ii][valid]
            extr = holo.utils.minmax(xx)
            extrema.append(extr)
            ee = np.linspace(*extr, nbins)
            edges.append(ee)
            axes[ii, ii].set(xlim=extr)
            axes[(npars - 1, ii)].set(xlabel=self.plt_labels[self.param_names[ii]])
            if ii > 0:
                axes[ii, 0].set(ylabel=self.plt_labels[self.param_names[ii]])

            for jj in range(ii):
                ax = axes[ii, jj]
                ax.set(ylim=extr)

        for (ii, jj), ax in axes.items():
            ax.grid(True, alpha=0.25)

            xx = self.params[:, jj][sort_idx[valid]]

            if jj > 0 and ii != jj:
                ax.tick_params(labelleft=False)

            if ii == jj:
                ax.hist(xx, histtype='step', bins=edges[jj], alpha=0.5, density=True, color='k', ls='--')
                ax.hist(xx, histtype='step', bins=edges[jj], weights=ww, alpha=0.5, density=True, color=colorsc[-1], lw=1.5)

                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                continue

            yy = self.params[:, ii][sort_idx[valid]]
            bins = (edges[jj], edges[ii])
            kale.contour((xx, yy), edges=bins, weights=ww, ax=ax, pad=0, smooth=1.5);
    
    def add_spectrum(self, ax, lw=3, errorbars=False, label=None):
        if not label:
            label = self.model_name
        vals = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sim_idx = np.argmax(vals)
        gwb = np.median(self.gwb, axis=2)
        ax.plot(np.log10(self.freqs), gwb[sim_idx], lw=lw, ls=self.line_style, c=self.color, label=label)
        if errorbars:
            valid = np.where((vals >= (np.nanmax(vals)- self.threshold)))[0]
            # print(len(valid) / len(vals))
            up = np.max(gwb[valid], axis=0)
            dn = np.min(gwb[valid], axis=0)
            ax.fill_between(np.log10(self.freqs), up, dn, color=self.color, alpha=0.25)

    def bhmf(self, mbh_log10, redz, fiducial=False):
        """
        Produce the black hole mass function at a given redshift.

        This is calculated using holodeck by convolving a double Schechter
        GSMF with an Mbh–M_bulge relation.

        Parameters
        ----------
        mass : tuple
            (min, max, npoints) in log10(Mbh/Msol), used as arguments
            to np.linspace().
        redz : float
            Redshift.
        fiducial : bool, optional
            Whether to use the fiducial values for the GSMF and MMBulge relation rather than the posterior values.
            If True, the fiducial values will be used, if False, the posterior values will be used. Default is False

        Returns
        -------
        masses : tuple
            Masses at which the BHMF is evaluated, in log10(Mbh/Msol).
        bhmf : array-like
            Black hole mass function at the given redshift.
        """
        if not fiducial:
            log10_phi1 = [self.posteriors['gsmf_log10_phi_one_z0'], self.posteriors['gsmf_log10_phi_one_z1'], self.posteriors['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.posteriors['gsmf_log10_phi_two_z0'], self.posteriors['gsmf_log10_phi_two_z1'], self.posteriors['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.posteriors['gsmf_log10_mstar_z0'], self.posteriors['gsmf_log10_mstar_z1'], self.posteriors['gsmf_log10_mstar_z2']]
            alpha1 = self.posteriors['gsmf_alpha_one']
            alpha2 = self.posteriors['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
            zplaw_amp = self.posteriors['mmb_zplaw'] if 'mmb_zplaw' in self.param_names else self.posteriors['mmb_zplaw_amp']
                
            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.posteriors['mmb_mamp_log10'],
                                                                    mplaw = self.posteriors['mmb_plaw'],
                                                                    zplaw_amp=zplaw_amp,
                                                                    zplaw_slope=self.posteriors['mmb_zplaw_slope'],
                                                                    zplaw_scatter=self.posteriors['mmb_zplaw_scatter'],
                                                                    scatter_dex = self.posteriors['mmb_scatter_dex'])
            
        if fiducial:
            log10_phi1 = [self.fiducial_values['gsmf_log10_phi_one_z0'], self.fiducial_values['gsmf_log10_phi_one_z1'], self.fiducial_values['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.fiducial_values['gsmf_log10_phi_two_z0'], self.fiducial_values['gsmf_log10_phi_two_z1'], self.fiducial_values['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.fiducial_values['gsmf_log10_mstar_z0'], self.fiducial_values['gsmf_log10_mstar_z1'], self.fiducial_values['gsmf_log10_mstar_z2']]
            alpha1 = self.fiducial_values['gsmf_alpha_one']
            alpha2 = self.fiducial_values['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.fiducial_values['mmb_mamp_log10'],
                                                                mplaw = self.fiducial_values['mmb_plaw'],
                                                                zplaw_amp=self.fiducial_values['mmb_zplaw_amp'],
                                                                zplaw_slope=self.fiducial_values['mmb_zplaw_slope'],
                                                                zplaw_scatter=self.fiducial_values['mmb_zplaw_scatter'],
                                                                scatter_dex = self.fiducial_values['mmb_scatter_dex'])
        
        return gsmf.mbh_mass_func_conv(10**mbh_log10 * MSOL, redz, mmbulge=mmb, scatter=True)
    
    def bhmf_err(self, mbh_log10, redz, ndraws=100):
        """
        Produces the black hole mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Parameters
        -----------
        mass : array
            black hole masses at which to evaluate the BHMF, in log10(Mbh/Msol)
        redz : float
            redshift

        Returns
        --------
        bhmf_median : array
            the median black hole mass function at the given redshift, calculated using the median posterior values for the parameters in the model
        bhmf_upper_bound : array
            the 84th percentile of the black hole mass function at the given redshift
        bhmf_lower_bound : array
            the 16th percentile of the black hole mass function at the given redshift
        TODO:
        * Add fiducial functionality
        """
        # masses = np.linspace(mass[0], mass[1], mass[2])
        
        self.bhmf_dict = copy.deepcopy(self.space_class.DEFAULTS)
        assert self.posteriors != self.fiducial_values, "Posteriors have not been calculated yet. Run get_posteriors() before calculating error bars on the BHMF."

        for par in self.bhmf_dict.keys():
            self.bhmf_dict[par] = np.array(np.random.normal(self.posteriors[par], scale=self.posteriors_err[par], size=ndraws))

        phis = []
        for j in range(ndraws):
            log10_phi1 = [self.bhmf_dict['gsmf_log10_phi_one_z0'][j], self.bhmf_dict['gsmf_log10_phi_one_z1'][j], self.bhmf_dict['gsmf_log10_phi_one_z2'][j]]
            log10_phi2 = [self.bhmf_dict['gsmf_log10_phi_two_z0'][j], self.bhmf_dict['gsmf_log10_phi_two_z1'][j], self.bhmf_dict['gsmf_log10_phi_two_z2'][j]]
            log10_mstar = [self.bhmf_dict['gsmf_log10_mstar_z0'][j], self.bhmf_dict['gsmf_log10_mstar_z1'][j], self.bhmf_dict['gsmf_log10_mstar_z2'][j]]
            alpha1 = self.bhmf_dict['gsmf_alpha_one'][j]
            alpha2 = self.bhmf_dict['gsmf_alpha_two'][j]

            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
            zplaw_amp = self.bhmf_dict['mmb_zplaw'][j] if 'mmb_zplaw' in self.param_names else self.bhmf_dict['mmb_zplaw_amp'][j]

            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.bhmf_dict['mmb_mamp_log10'][j],
                                                            mplaw = self.bhmf_dict['mmb_plaw'][j],
                                                            zplaw_amp=zplaw_amp,
                                                            zplaw_slope=self.bhmf_dict['mmb_zplaw_slope'][j],
                                                            zplaw_scatter=self.bhmf_dict['mmb_zplaw_scatter'][j],
                                                            scatter_dex = self.bhmf_dict['mmb_scatter_dex'][j])
            
            phis.append(gsmf.mbh_mass_func_conv(10**mbh_log10 * MSOL, redz, mmbulge=mmb, scatter=True))

        phi_50, phi_84, phi_16 = np.nanpercentile(phis, [50, 84, 16], axis = 0)
        
        return phi_50, phi_84, phi_16
    
    def gsmf(self, mstar_log10, redz, fiducial=False):
        """
        Produces the galaxy stellar mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Parameters
        -----------
        mass: tuple (min, max, npoints) in log10(Msun/Msol) to be used as arguments in np.linspace()
        redz: redshift
        
        Returns
        --------
        The galaxy stellar mass function at the given redshift
        """

        if not fiducial:
            log10_phi1 = [self.posteriors['gsmf_log10_phi_one_z0'], self.posteriors['gsmf_log10_phi_one_z1'], self.posteriors['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.posteriors['gsmf_log10_phi_two_z0'], self.posteriors['gsmf_log10_phi_two_z1'], self.posteriors['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.posteriors['gsmf_log10_mstar_z0'], self.posteriors['gsmf_log10_mstar_z1'], self.posteriors['gsmf_log10_mstar_z2']]
            alpha1 = self.posteriors['gsmf_alpha_one']
            alpha2 = self.posteriors['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
        if fiducial:
            log10_phi1 = [self.fiducial_values['gsmf_log10_phi_one_z0'], self.fiducial_values['gsmf_log10_phi_one_z1'], self.fiducial_values['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.fiducial_values['gsmf_log10_phi_two_z0'], self.fiducial_values['gsmf_log10_phi_two_z1'], self.fiducial_values['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.fiducial_values['gsmf_log10_mstar_z0'], self.fiducial_values['gsmf_log10_mstar_z1'], self.fiducial_values['gsmf_log10_mstar_z2']]
            alpha1 = self.fiducial_values['gsmf_alpha_one']
            alpha2 = self.fiducial_values['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
        
        return gsmf(10**mstar_log10 * MSOL, redz)
    

    def get_shenf(self, redshift, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Retrieve the fit to the Shen+2020 bolometric luminosity function at a given redshift.
        
        Parameters
        -----------  
        redshift : flaot
            redshift of the fit to be used 0.2-7.0 in steps of 0.2
        path_to_shen_fits : str, optional
            path to the Shen+2020 fits for the bolometric LF at different redshifts. Default is "/Users/cayenne/Documents/Research/quasarlf/qlffits/"


        Returns
        --------
        x : array
            the x-axis values, luminosity in log10(L/erg/s)
        y : array
            fit to log phiL
        """
        dat = np.genfromtxt(path_to_shen_fits+"bolometric_fit_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']

    def get_shend(self, redshift, path_to_shen_data="/Users/cayenne/Documents/Research/quasarlf/qlfdata/"):
        """
        Retrieve the data from the Shen+2020 bolometric luminosity function at a given redshift.

        Parameters
        -----------  
        redshift : flaot
            redshift of the fit to be used 0.2-7.0 in steps of 0.2
        path_to_shen_fits : str, optional
            path to the Shen+2020 fits for the bolometric LF at different redshifts. Default is "/Users/cayenne/Documents/Research/quasarlf/qlfdata/"

        Returns
        --------
        x : array
            the x-axis values, luminosity in log10(L/erg/s)
        y : array
            data for log phiL
        """
        dat = np.genfromtxt(path_to_shen_data+"bolometric_data_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']

    def calculate_radiative_efficiency(self, zval, step=1e-3, mass=[5, 13, 100], fiducial=False, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Calculate the radiative efficiency implied by the model at a given redshift by comparing the change in the black hole mass function between two redshifts to the
        luminosity function at the average redshift. This is a rough calculation that assumes that the change in the BHMF and LF between the two redshifts is solely due to accretion
        and does not consider mergers or other processes that may contribute to the growth of black holes.

        May have bugs, shouldn't be used

        Parameters
        -----------
        zval : float
            the redshift at which to calculate the radiative efficiency
        step : float, optional
            the step in redshift to use for calculating the change in the BHMF and LF. Default is 1e-3
        fiducial : bool, optional
            whether to use the fiducial values of the model parameters instead of the posteriors. Default is False
        path_to_shen_fits: str, optional
            path to the Shen+2020 fits for the bolometric LF at different redshifts. Default is "/Users/cayenne/Documents/Research/quasarlf/qlffits/"

        Returns
        --------
        erad : float
            radiative efficiency between the two redshifts
        mdot : float
            the total integrated mass density gain between those redshifts (scaled)
        Lum : float
            the total integrated luminosity at the latter redshift

        Issues:

        * Incorporate AGN Fraction
        * Radiative efficiency may be mass and redshift dependent, not currently implemented
        * Need to make sure that it is always integrating over roughly similar mass - luminosity ranges
        * Integration only cosiders two bins and nothing in between, but should be fine for order of magnitude calculation
        * May have a volume normalization issue

        """
        f_obsc = 1/3  # Fraction of AGN that are not obscured and therefore observed in the LF
        f_acc = 0.9  # Fraction of BH growth due to accretion as opposed to mergers
        # Get BHMF at each redshift
        # volume = cosmo.comoving_volume(z2) - cosmo.comoving_volume(z1)

        z1 = zval
        z2 = zval + step

        dt = cosmo.lookback_time(z2) - cosmo.lookback_time(z1)

        # Mass Function
        masses, bhmf1 = self.bhmf(mass, redz=z1, fiducial=fiducial)
        masses, bhmf2 = self.bhmf(mass, redz=z2, fiducial=fiducial)
        
        mdot = trapz((bhmf1 - bhmf2) * 10**masses, masses) * u.Msun / dt * f_obsc / f_acc#/ u.Mpc**3 * volume

        if mdot < 0:
            print('Warning: Negative mass accreted between z = {} and z = {} for model {}.'.format(np.round(z1, 2), np.round(z2, 2), self.model_name))

        # Luminosity Function

        shen_fit = self.get_shenf(zval, path_to_shen_fits)
        phiL = 10**(shen_fit[1])
        logL = shen_fit[0]

        Lum = trapz((phiL) * 10**logL, logL) * u.erg / u.s #/ u.Mpc**3 * volume

        # Radiative Efficiency Calculation

        erad = Lum / (mdot * c.c**2)
        # erad = mdot

        return erad.decompose(), mdot.to(u.Msun / u.yr), Lum
    
    def fdfunc(self, mbh_log10, redshift, fiducial=False, fdmin=0.06):
        """
        Fit from https://ui.adsabs.harvard.edu/abs/2024ApJ...964..183Z/graphics
        """
        norm_fit = [-0.0348215, 0.77511731, -4.24506371]  # [-0.25916918189249905, 5.489176958654701, -31.25532992258093]
        slope_fit = [ 0.01273298, -0.28087742, 1.5361624 ]  # [0.2927610944131739, -6.537700910036786, 35.65876956759064]

        if fiducial:
            zplaw_amp = self.fiducial_values['mmb_zplaw_amp']
            mamp_z = 10**self.fiducial_values['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp
            mplaw_z = self.fiducial_values['mmb_plaw'] * (1.0 + redshift)**self.fiducial_values['mmb_zplaw_slope']

        if not fiducial:
            zplaw_amp = self.posteriors['mmb_zplaw'] if 'mmb_zplaw' in self.param_names else self.posteriors['mmb_zplaw_amp']
            mamp_z = 10**self.posteriors['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp
            mplaw_z = self.posteriors['mmb_plaw'] * (1.0 + redshift)**self.posteriors['mmb_zplaw_slope']

        mstar_log10 = (mbh_log10 - np.log10(mamp_z)) / mplaw_z + 11

        norm = norm_fit[0]*mstar_log10**2 + norm_fit[1]*mstar_log10 + norm_fit[2]
        norm[norm < 0.0] = 0.0
        slope = slope_fit[0]*mstar_log10**2 + slope_fit[1]*mstar_log10 + slope_fit[2]
        slope[slope > 0.0] = 0.0

        phi_fd = norm * (redshift) + slope

        phi_fd[phi_fd < fdmin] = fdmin

        return phi_fd
    
    def bhmf_from_gsmf(self, mstar, mbh, redshift, fiducial=False):
        """
        Like bhmf_conv in holodeck except this starts with a GSMF and the convolution is done via dot product
        """

        mstar_log10, ndens = self.gsmf([mstar[0], mstar[1], mstar[2]], redz=redshift, fiducial=True)
        mbh_log10  = np.linspace(mbh[0], mbh[1], mbh[2])

        if fiducial:
            scatter = np.log10(10**self.fiducial_values['mmb_scatter_dex'] * (1.0 + redshift)**self.fiducial_values['mmb_zplaw_scatter'])

            zplaw_amp = self.fiducial_values['mmb_zplaw_amp']
            mamp_z = 10**self.fiducial_values['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp

            mplaw_z = self.fiducial_values['mmb_plaw'] * (1.0 + redshift)**self.fiducial_values['mmb_zplaw_slope']
        
        if not fiducial:
            scatter = np.log10(10**self.posteriors['mmb_scatter_dex'] * (1.0 + redshift)**self.posteriors['mmb_zplaw_scatter'])

            zplaw_amp = self.posteriors['mmb_zplaw'] if 'mmb_zplaw' in self.param_names else self.posteriors['mmb_zplaw_amp']
            mamp_z = 10**self.posteriors['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp

            mplaw_z = self.posteriors['mmb_plaw'] * (1.0 + redshift)**self.posteriors['mmb_zplaw_slope']

        logMbh_mean = np.log10(mamp_z) + mplaw_z * (mstar_log10 - 11.0)

        inv_sqrt2pi = 1.0 / np.sqrt(2*np.pi)
        K = inv_sqrt2pi/scatter * np.exp( -0.5*((mbh_log10[:, None] - logMbh_mean)/scatter)**2)

        dlogM = mbh_log10[1] - mbh_log10[0]
        bhmf_conv = np.dot(K, ndens) * dlogM

        return mbh_log10, bhmf_conv
    
    def bhargal(self, mbh_log10, redshift, fiducial=False):
        """
        Calculate black hole accretion rate as a function of black hole mass and redshift using the MMBulge relation and the GSMF.

        The function in the paper is in terms of stellar mass, but we can use the MMBulge relation to convert it to a function of black hole mass.
        Fit from https://ui.adsabs.harvard.edu/abs/2024ApJ...964..183Z/graphics
        Msun / year
        Error ranges from 0.1 - 0.3 dex depending on redshift and mass (see Figure 6)
        """
        if fiducial:
            zplaw_amp = self.fiducial_values['mmb_zplaw_amp']
            mamp_z = 10**self.fiducial_values['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp
            mplaw_z = self.fiducial_values['mmb_plaw'] * (1.0 + redshift)**self.fiducial_values['mmb_zplaw_slope']

        if not fiducial:
            zplaw_amp = self.posteriors['mmb_zplaw'] if 'mmb_zplaw' in self.param_names else self.posteriors['mmb_zplaw_amp']
            mamp_z = 10**self.posteriors['mmb_mamp_log10'] * (1.0 + redshift)**zplaw_amp
            mplaw_z = self.posteriors['mmb_plaw'] * (1.0 + redshift)**self.posteriors['mmb_zplaw_slope']

        mstar_log10 = (mbh_log10 - np.log10(mamp_z)) / mplaw_z + 11

        c, k, b = 2.53850958, 0.85309541, -18.35185436
        intercept = c * (1 - np.exp(-k * redshift)) + b

        return 1.3595507359218555 * mstar_log10 + intercept
    
    def eta_from_mbh_davis(self, mbh_log10):
        """
        https://iopscience.iop.org/article/10.1088/0004-637X/728/2/98/pdf

        Not redshift dependent
        """

        etas = 0.089 * (10**mbh_log10/ 1e8)**0.52
        etas[etas > 1] = 1
        return etas

    def eta_from_mbh_line_slow(self, mbh_log10):
        """
        https://ui.adsabs.harvard.edu/abs/2012ApJ...749..187L/abstract
        Does not change with redshift, not advised to use for redshifts below 0.8. Two lines of different slopes with a cutoff
        """

        m = 0.24675530275901314
        b = -1.4300561796413318

        eta_high = m*mbh_log10 + b

        y1 = 0
        y2 = eta_high[mbh_log10 <= 7][-1]

        x1 = 1
        x2 = 7
        eta_log = (y2 - y1) / (x2 - x1) * (mbh_log10[mbh_log10 <= 7] - x1) + y1
        etas = np.append(eta_log, eta_high[mbh_log10 > 7])

        etas[etas > 1] = 1
        return etas
    
    def eta_from_mbh_line_extra_slow(self, mbh_log10):
        """
        https://ui.adsabs.harvard.edu/abs/2012ApJ...749..187L/abstract
        Not advised to use for redshifts below 0.8. Two lines of different slopes with a cutoff
        """

        m = 0.24675530275901314
        b = -1.4300561796413318

        eta_high = m*mbh_log10 + b

        msklo = np.where(mbh_log10 <= 7)[0]
        mskhi = np.where(mbh_log10 > 7)[0]

        y1 = 0
        y2 = eta_high[msklo][-1]

        x1 = 1
        x2 = 7
        eta_log = (y2 - y1) / (x2 - x1) * (mbh_log10[msklo] - x1) + y1
        etas = np.append(eta_log, eta_high[mskhi])

        etas[etas > 1] = 1
        return etas
    
    def eta_from_mbh_line(self, mbh_log10):
        """
        https://ui.adsabs.harvard.edu/abs/2012ApJ...749..187L/abstract
        Not advised to use for redshifts below 0.8. Two lines of different slopes with a cutoff
        """

        m = 0.24675530275901314
        b = -1.4300561796413318

        etas = m*mbh_log10 + b

        y1 = 0
        y2 = etas[mbh_log10 <= 7][-1]

        x1 = 1
        x2 = 7
        etas[mbh_log10 <= 7] = (y2 - y1) / (x2 - x1) * (mbh_log10[mbh_log10 <= 7] - x1) + y1

        etas[etas > 1] = 1
        return etas

    def eta_from_mbh_logistic(self, mbh_log10, min=0.25, k=-1.4, m0=8.4):
        """
        Not advised to use for redshifts below 0.8. Logistic function
        """

        l = 1.0 - min
        return l / (1 + np.exp(k * (mbh_log10 - m0))) + min
    

    def L_from_Mbh_via_mdot_eta_func(self, mbh_log10, lums_log10, redshift, fiducial=False, eta_func = 'Davis', rad_eff=None):
        """
        Calculate luminosity from black hole mass using the accretion rate and radiative efficiency.
        The accretion rate is calculated using the MMBulge relation and the GSMF, and the radiative efficiency is calculated using one of several functions of black hole mass.

        Parameters
        -----------
        mbh_log10 : tuple
            black hole masses to evaluate the luminosity function at, in log10(Mbh/Msol)
        lums_log10 : array
            array of log10 luminosities at which to evaluate the luminosity function
        redshift : float
            redshift at which to evaluate the luminosity function
        fiducial : bool, optional
            whether to use the fiducial values of the model parameters instead of the posteriors. Default is False
        eta_func : bool, optional
            which functional form to use for calculating radiative efficiency, options are 'Davis', 'Logistic', 'Line', and 'Constant'. Default is 'Davis'
        rad_eff : float, optional
            the constantvalue of the radiative efficiency to use when eta_func is 'Constant'. Default is None

        Returns
        --------
        lf_conv: the luminosity function calculated from the black hole mass function, accretion rate, and radiative efficiency
        """
        scattereta = 0.5
        scattermdotmstar = 0.3

        if fiducial:
            scattermmb = np.log10(10**self.fiducial_values['mmb_scatter_dex'] * (1.0 + redshift)**self.fiducial_values['mmb_zplaw_scatter'])
            scatter = np.sqrt(scattermmb**2 + scattereta**2 + scattermdotmstar**2)

        if not fiducial:
            scattermmb = np.log10(10**self.posteriors['mmb_scatter_dex'] * (1.0 + redshift)**self.posteriors['mmb_zplaw_scatter'])
            scatter = np.sqrt(scattermmb**2 + scattereta**2 + scattermdotmstar**2)

        ndens = self.bhmf(mbh_log10, redshift, fiducial=fiducial)

        Mdot_mean = 10**self.bhargal(mbh_log10, redshift, fiducial=fiducial) * 6.3008906592961785e+25  # Msun / year to g / s

        if eta_func == 'Davis':
            etas = self.eta_from_mbh_davis(mbh_log10)

        elif eta_func == 'Logistic':
            etas = self.eta_from_mbh_logistic(mbh_log10)
        
        elif eta_func == 'Line':
            etas = self.eta_from_mbh_line(mbh_log10)
        
        elif eta_func == 'Constant':
            if rad_eff:
                etas = rad_eff
            else:
                raise ValueError("Please provide a radiative efficiency value.")
        
        Lmean_log10 = np.log10(Mdot_mean * (2.9979246e10)**2 * etas)

        inv_sqrt2pi = 1.0 / np.sqrt(2*np.pi)
        K = inv_sqrt2pi/scatter * np.exp( -0.5*((lums_log10[:, None] - Lmean_log10)/scatter)**2)

        dlogL = lums_log10[1] - lums_log10[0]
        lf_conv = np.dot(K, ndens) * dlogL

        return lf_conv