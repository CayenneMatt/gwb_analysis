"""
Class to aid in GWB model analysis
"""
import numpy as np
import scipy.stats as stats

import holodeck as holo
import holodeck.sams as sams
from holodeck.constants import MSOL, NWTG, KMPERSEC
from holodeck import librarian

import kalepy as kale
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
try:
    from numpy import trapz
except:
    from numpy import trapezoid as trapz
import astropy.units as u
import astropy.constants as c
import scipy.integrate as inte

import ast

import copy

C_edd = (4 * np.pi * c.G * c.u * c.c / c.sigma_T).to(u.erg / u.s / u.Msun).value # erg/s per Msun

class Model_Info(object):
    """
    Class to analyze gravitational wave background models generated using holodeck.
    This class automatically reads the data and assigns attributes for the GWB amplitudes, parameters, likelihoods, and frequencies.

    .. seealso::
        This class works with outputs from `holodeck <https://github.com/nanograv/holodeck>`_.
    
    Parameters
    -----------
    path : str
        Path to the data, should end with a '/' and be the same as the path used to generate the data
    file : str
        Filename of the data, should be the same as the filename used to generate the data
    model_name :  str
        Name of the model, used for plotting
    color : str
        Color for all plotting associated with this model
    line_style : str
        Line style for all plotting associated with this model

    threshold : float, optional
        Minimum likelihood cutoff, used for plotting error bars on the spectrum. Default is 0.5
    evolving : bool, optional
        Whether the model has MMBulge evolution, used for plotting. Default is False
    stdev : float, optional
        Standard deviation of evolution parameter, used for plotting, should be None if evolving is False. Default is None
    nfreq : int, optional
        Number of frequency bins to use in likelihood calculation, 5 is recommended. Default is 5
    param_space_name : str, optional
        Name of parameter space, should be the same as the parameter space used to generate the data. Default is 'PS_Astro_Strong_All'

    Attributes
    -----------
    path : str
        Path to the data specified as an argument
    file : str
        Filename of the data specified as an argument
    param_space_name : str
        Name of parameter space specified as an argument
    gwb : array-like
        GWB amplitudes for each model, shape is (nsamples, nrealizations, nfreqs)
    param_names : array-like
        Names of parameters in the model
    params : dictionary
        Dictionary of median posterior values for each parameter in the model, will be the same as fiducial unless get_posteriors() is called
    param_samples : array-like
        Parameter sample associated with each model
    ln_like : array-like
        Log-likelihood of each model
    freqs : array-like
        Frequencies at which the amplitudes are calculated
    model_name : str
        Model name specified as an argument
    color :  str
        Color for all plotting associated with this model specified as an argument
    line_style : str
        Line style for all plotting associated with this model specified as an argument
    threshold : float
        minimum likelihood cutoff specified as an argument
    evolving : bool
        Whether the model has MMBulge evolution specified as an argument
    stdev : float
        Standard deviation of evolution parameter specified as an argument
    nfreq : int
        Number of frequency bins to use in likelihood calculation specified as an argument
    idcs : array-like
        Index of evolution parameter, None if evolving is False
    space_class : class
        Class of the parameter space, used to get priors and fiducial values
    fiducial_values:  dict
        Dictionary of fiducial values for each parameter in the model
    params : dict
        Dictionary of median posterior values for each parameter in the model,
        will be the same as fiducial unless get_posteriors() is called
    params_err : dict
        Dictionary of standard deviation of posterior values for each parameter in the model,
        will be the same as fiducial values unless get_posteriors() is called
    plt_labels : dict
        Dictionary of plot labels associated with each parameter, used for plotting
    """
    # Attributes
    def __init__(self, path, file, model_name, color, line_style, threshold=0.5, evolving=False, stdev=None, nfreq=5, param_space_name='PS_Astro_Strong_All'):

        self.path = path  # Path to data
        self.file = file  # Filename
        self.param_space_name = param_space_name  # Name of parameter space

        # Data
        dat = np.load(self.path+self.file)
        self.gwb = dat['gwb']  # GWB amplitudes for each model
        try:
            self.param_names = dat['names']  # Names of parameters in the model
        except:
            self.param_names = np.array(ast.literal_eval(input('Parameter names not found in data, please input parameter names as a list: ')))
        self.param_samples = dat['params']  # Parameter sample associated with each model
        self.ln_like = dat['ln_like']  # Log-likelihood of each model
        self.freqs = np.array(dat['fobs_cents'])  # Frequencies for all models

        # Plotting and book keeping
        self.model_name = model_name  # Model name
        self.color = color  # Color for all plotting associated with this model
        self.line_style = line_style  # Line style for all plotting associated with this model
        self.threshold = threshold  # Offset from the maximum likelihood to use as a cutoff for plotting error bars on the spectrum
        self.evolving = evolving  # Whether the model has MMBulge evolution
        self.stdev = stdev  # Standard deviation of evolution parameter
        self.nfreq = nfreq  # Number of frequency bits to fit to. Default is 5
        self.idcs = None  # Index of evolution parameter

        try:
            id = np.where((self.param_names == 'mmb_zplaw') | (self.param_names == 'mmb_zplaw_amp') | (self.param_names == 'mmb_zplaw_slope') | (self.param_names == 'mmb_zplaw_scatter'))[0]
            if len(id) > 0:
                self.idcs = id
        except:
            self.idcs = None

        if 'mmb_zplaw' in self.param_names:
            self.param_names[id] = 'mmb_zplaw_amp'
        
        # Dictionary of priors to be modified by get_posteriors(), note that LM* models have different priors, but currently sample every parameter.
        self.space_class = librarian.param_spaces_dict[self.param_space_name]
        self.fiducial_values = copy.deepcopy(self.space_class.DEFAULTS)
        self.params = copy.deepcopy(self.space_class.DEFAULTS)

        self.fiducial_values_err = {'hard_time': 3,
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
        
        self.params_err = copy.deepcopy(self.fiducial_values_err)

        # Dictionary of plot labels associated with each parameter
        self.plt_labels = {'hard_time'              : r"$\tau_\mathrm{f}$",
                            'hard_sepa_init'        : r"Bin. Sep.",
                            'hard_rchar'            : r'R$_\mathrm{char}$',
                            'hard_rchar_9'            : r'R$_\mathrm{char}$',
                            'hard_gamma_inner'      : r"$\nu_\mathrm{inner}$",
                            'hard_gamma_outer'      : r"$\nu_\mathrm{outer}$",
                            'hard_outer_time'       : r"$\tau_\mathrm{outer}$",
                            'hard_nu_inner'         : r"$\nu_\mathrm{inner}$",
                            'hard_r_gw_crit_9'      : r'R$_\mathrm{GW, crit}$',
                            'hard_alpha_gw_crit'    : r"$\alpha_\mathrm{GW, crit}$",
                            'hard_beta_gw_crit'     : r"$\beta_\mathrm{GW, crit}$",
                            'gsmf_phi0_log10'       : r"$\log \phi_{*}$",
                            'gsmf_mchar0_log10'     : r"$\log M_{\mathrm{c}}$",
                            'gsmf_phi0_p'             : r"GSMF $\psi_0$",
                            'gsmf_mchar0_log10_p'     : r"GSMF $m_{\psi,0}$",
                            'mmb_mamp_log10_p'        : r"MMB $\mu$",
                            'mmb_scatter_dex_p'       : r"MMB $\epsilon_{\mu}$",
                            'hard_time_p'             : r"phenom $\tau_f$",
                            'hard_gamma_inner_p'      : r"phenom $\nu_\mathrm{inner}$",
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
        Returns parameter names and sampled distributions from the prior parameter space
        
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
            xx = self.param_samples[:, jj]
            xx = xx[valid]
            rv = stats.rv_histogram(np.histogram(xx, weights=weights_med))
            median = rv.median()
            stdev = rv.std()
            self.params[n] = median
            self.params_err[n] = stdev

        return self

    def plot_histogram(self, ax, param_name, nbins=20, prior=False, **kwargs):
        """
        Plot a histogram of the posterior distribution for a given parameter, with optional prior distribution overlaid.
        
        Parameters
        ----------
        ax : matplotlib axis
            axis to which the histogram will be added
        param_name : str
            name of the parameter to plot, should be one of the parameter names in the model
        nbins : int, optional
            number of bins to use in the histogram, default is 20
        label : str, optional
            label for the histogram, default is None, in which case the parameter name will be used
        prior : bool, optional
            whether to plot the prior distribution overlaid on the histogram, default is False
        
        Returns
        -------
        self, the histogram is added to the given axis
        """
        skip = None

        gwb_med = np.median(self.gwb, axis=-1)
        valid = np.any(gwb_med > 0, axis=1)

        like_med = np.sum(self.ln_like[::skip, :self.nfreq], axis=1)
        weights_med = np.exp(like_med[valid])

        msk = np.where(self.param_names == param_name)[0]

        xx = self.param_samples[:, msk]
        xx = xx[valid]
        if prior:
            ax.hist(xx, density=True, bins=nbins, **kwargs)
        if not prior:
            ax.hist(xx, weights=weights_med, density=True, bins=nbins, **kwargs)

        return self
    
    def corner_plot(self, fontsize=25, nbins=20, cmap='Blues'):
        """
        Old and slow, but tested version of corner plot, will be removed in favor of corner_plot_fast() pending appropriate testing. Creates a corner plot for all parameters in the model.

        Parameters
        ----------
        nbins : int, optional
            Number of bins to use in the histograms and contour plots. Default is 20
        cmap : str, optional
            Name of matplotlib colormap to use for the points in the corner plot. Default is 'Blues'

        Returns
        -------
        None, the corner plot is displayed using matplotlib
        """
        npars = len(self.param_names)
        nsamp, npars = self.param_samples.shape

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
            xx = self.param_samples[:, ii][valid]
            extr = holo.utils.minmax(xx)
            extrema.append(extr)
            ee = np.linspace(*extr, nbins)
            edges.append(ee)
            axes[ii, ii].set(xlim=extr)
            axes[-1, ii].set_xlabel(self.plt_labels[self.param_names[ii]], fontsize=fontsize)
            if ii > 0:
                axes[ii, 0].set_ylabel(self.plt_labels[self.param_names[ii]], fontsize=fontsize)
            # axes[-1, ii].set(xlabel=self.plt_labels[self.param_names[ii]])
            # if ii > 0:
            #     axes[ii, 0].set(ylabel=self.plt_labels[self.param_names[ii]])
            for jj in range(ii):
                ax = axes[ii, jj]
                ax.set(ylim=extr)

        for (ii, jj), ax in np.ndenumerate(axes):
            ax.tick_params(axis='both', labelsize=fontsize-5)
            if jj > ii:
                ax.set_visible(False)
                continue
            ax.grid(True, alpha=0.25)
            xx = self.param_samples[:, jj][sort_idx[valid]]

            if jj > 0 and ii != jj:
                ax.set_yticklabels(['' for ii in ax.get_yticks()])

            if ii == jj:
                ax.hist(xx, histtype='step', bins=edges[jj], alpha=0.5, density=True, color='k', ls='--')
                ax.hist(xx, histtype='step', bins=edges[jj], weights=ww, alpha=0.5, density=True, color=colorsc[-1], lw=1.5)

                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                continue

            yy = self.param_samples[:, ii][sort_idx[valid]]
            bins = (edges[jj], edges[ii])
            kale.contour((xx, yy), edges=bins, weights=ww, ax=ax, pad=0, smooth=1.5);
        for ax in axes[-1, :]:
            ax.tick_params(axis='x', labelrotation=45)

        fig.subplots_adjust(wspace=0.1, hspace=0.1);


    def corner_plot_fast(self, nbins=20, cmap='Blues'):
        """
        Faster version of corner plot. Creates a corner plot for all parameters in the model.

        Parameters
        ----------
        nbins : int, optional
            Number of bins to use in the histograms and contour plots. Default is 20
        cmap : str, optional
            Name of matplotlib colormap to use for the points in the corner plot. Default is 'Blues'

        Returns
        -------
        None, the corner plot is displayed using matplotlib
        """

        npars = len(self.param_names)
        nsamp, npars = self.param_samples.shape

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
            xx = self.param_samples[:, ii][valid]
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

            xx = self.param_samples[:, jj][sort_idx[valid]]

            if jj > 0 and ii != jj:
                ax.tick_params(labelleft=False)

            if ii == jj:
                ax.hist(xx, histtype='step', bins=edges[jj], alpha=0.5, density=True, color='k', ls='--')
                ax.hist(xx, histtype='step', bins=edges[jj], weights=ww, alpha=0.5, density=True, color=colorsc[-1], lw=1.5)

                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                continue

            yy = self.param_samples[:, ii][sort_idx[valid]]
            bins = (edges[jj], edges[ii])
            kale.contour((xx, yy), edges=bins, weights=ww, ax=ax, pad=0, smooth=1.5);
    
    def add_spectrum(self, ax, line=True, errorbars=False, logx=True, logy=False, label=None, **kwargs):
        """
        Add the spectrum of the model to a given axis, with optional error bars and label.

        Parameters
        ----------
        ax : Matplotlib axis
            axis to which the spectrum will be added
        line : bool, optional
            Whether to add a line for the spectrum. Default is True
        errorbars : bool, optional
            Whether to add error bars to the spectrum. Default is False
        logx : bool, optional
            Whether to plot the spectrum with a logarithmic x-axis. Default is True
        logy : bool, optional
            Whether to plot the spectrum with a logarithmic y-axis. Default is False
        label : str, optional
            Label for the spectrum. Default is None, in which case the model name will be used
        kwargs :
            additional keyword arguments to be passed to the plot function, such as linewidth or alpha

        Returns
        -------
        None, the spectrum is added to the given axis
        """
        if not label:
            label = self.model_name
        vals = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sim_idx = np.argmax(vals)
        
        gwb = np.median(self.gwb, axis=2)

        if logx == True:
            freqs = np.log10(self.freqs)
        else:
            freqs = self.freqs

        if line == True:
            if logy == True:
                gwby = np.log10(gwb[sim_idx])
            else:
                gwby = gwb[sim_idx]

            ax.plot(freqs, gwby, ls=self.line_style, c=self.color, label=label, **kwargs)

        if errorbars:
            valid = np.where((vals >= (np.nanmax(vals)- self.threshold)))[0]
            up = np.max(gwb[valid], axis=0)
            dn = np.min(gwb[valid], axis=0)
            if logy == True:
                up = np.log10(up)
                dn = np.log10(dn)

            ax.fill_between(freqs, up, dn, color=self.color, alpha=0.25)

    def bhmf(self, mbh_log10, redshift):
        r"""
        Produce the black hole mass function at a given redshift. This is calculated using holodeck by convolving a double Schechter GSMF with an Mbh–M_bulge relation
        :math:`M_\mathrm{BH} = \alpha_0 \left(\frac{M_{\mathrm{bulge}}}{10^{11}\, M_{\odot}}\right)^{\beta_0}`

        Parameters
        ----------
        mbh_log10 : array
            Black hole masses at which the BHMF is evaluated, in log10(Mbh/Msol)
        redshift : float
            Redshift at which the BHMF is evaluated

        Returns
        -------
        bhmf : array-like
            Black hole mass function at the given redshift
        """
        try:
            log10_phi1 = [self.params['gsmf_log10_phi_one_z0'], self.params['gsmf_log10_phi_one_z1'], self.params['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.params['gsmf_log10_phi_two_z0'], self.params['gsmf_log10_phi_two_z1'], self.params['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.params['gsmf_log10_mstar_z0'], self.params['gsmf_log10_mstar_z1'], self.params['gsmf_log10_mstar_z2']]
            alpha1 = self.params['gsmf_alpha_one']
            alpha2 = self.params['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)

        except:
            log10_phi1 = self.params['gsmf_phi0_log10']
            log10_mstar = self.params['gsmf_mchar0_log10']

            gsmf = sams.GSMF_Schechter(phi0=log10_phi1, mchar0_log10=log10_mstar)

        try:
            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.params['mmb_mamp_log10'],
                                                            mplaw = self.params['mmb_plaw'],
                                                            zplaw_amp = self.params['mmb_zplaw_amp'],
                                                            zplaw_slope = self.params['mmb_zplaw_slope'],
                                                            zplaw_scatter = self.params['mmb_zplaw_scatter'],
                                                            scatter_dex = self.params['mmb_scatter_dex'])
        except:
            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.params['mmb_mamp_log10'],
                                                            mplaw = self.params['mmb_plaw'],
                                                            scatter_dex = self.params['mmb_scatter_dex'])

        
        return gsmf.mbh_mass_func_conv(10**mbh_log10 * MSOL, redshift, mmbulge=mmb, scatter=True)
    
    def bhmf_err(self, mbh_log10, redshift, ndraws=100):
        """
        Produces the black hole mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Parameters
        -----------
        mbh_log10 : array
            Black hole masses at which to evaluate the BHMF, in log10(Mbh/Msol)
        redshift : float
            Redshift at which the BHMF is evaluated

        Returns
        --------
        bhmf_median : array
            The median black hole mass function at the given redshift, calculated using the median posterior values for the parameters in the model
        bhmf_upper_bound : array
            The 84th percentile of the black hole mass function at the given redshift
        bhmf_lower_bound : array
            The 16th percentile of the black hole mass function at the given redshift
        """        
        self.bhmf_dict = copy.deepcopy(self.space_class.DEFAULTS)

        # assert self.params != self.fiducial_values, "Posteriors have not been calculated yet. Run get_posteriors() before calculating error bars on the BHMF."

        for par in self.bhmf_dict.keys():
            self.bhmf_dict[par] = np.array(np.random.normal(self.params[par], scale=self.params_err[par], size=ndraws))

        phis = []
        for j in range(ndraws):
            try:
                log10_phi1 = [self.bhmf_dict['gsmf_log10_phi_one_z0'][j], self.bhmf_dict['gsmf_log10_phi_one_z1'][j], self.bhmf_dict['gsmf_log10_phi_one_z2'][j]]
                log10_phi2 = [self.bhmf_dict['gsmf_log10_phi_two_z0'][j], self.bhmf_dict['gsmf_log10_phi_two_z1'][j], self.bhmf_dict['gsmf_log10_phi_two_z2'][j]]
                log10_mstar = [self.bhmf_dict['gsmf_log10_mstar_z0'][j], self.bhmf_dict['gsmf_log10_mstar_z1'][j], self.bhmf_dict['gsmf_log10_mstar_z2'][j]]
                alpha1 = self.bhmf_dict['gsmf_alpha_one'][j]
                alpha2 = self.bhmf_dict['gsmf_alpha_two'][j]

                gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            except:
                log10_phi1 = self.bhmf_dict['gsmf_phi0_log10']
                log10_mstar = self.bhmf_dict['gsmf_mchar0_log10']

                gsmf = sams.GSMF_Schechter(phi0=log10_phi1, mchar0_log10=log10_mstar)

            try:
                mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.bhmf_dict['mmb_mamp_log10'][j],
                                                                mplaw = self.bhmf_dict['mmb_plaw'][j],
                                                                zplaw_amp=self.bhmf_dict['mmb_zplaw_amp'][j],
                                                                zplaw_slope=self.bhmf_dict['mmb_zplaw_slope'][j],
                                                                zplaw_scatter=self.bhmf_dict['mmb_zplaw_scatter'][j],
                                                                scatter_dex = self.bhmf_dict['mmb_scatter_dex'][j])
            except:
                mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.bhmf_dict['mmb_mamp_log10'][j],
                                                            mplaw = self.bhmf_dict['mmb_plaw'][j],
                                                            scatter_dex = self.bhmf_dict['mmb_scatter_dex'][j])
            
            phis.append(gsmf.mbh_mass_func_conv(10**mbh_log10 * MSOL, redshift, mmbulge=mmb, scatter=True))

        phi_50, phi_84, phi_16 = np.nanpercentile(phis, [50, 84, 16], axis = 0)
        
        return phi_50, phi_84, phi_16
    
    def gsmf(self, mstar_log10, redshift):
        """
        Produces the galaxy stellar mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Parameters
        -----------
        mstar_log10 : array
            Black hole masses at which to evaluate the BHMF, in log10(Mbh/Msol)
        redshift : float
            Redshift at which the BHMF is evaluated
        
        Returns
        --------
        gsmf : array
            The galaxy stellar mass function at the given redshift
        """

        log10_phi1 = [self.params['gsmf_log10_phi_one_z0'], self.params['gsmf_log10_phi_one_z1'], self.params['gsmf_log10_phi_one_z2']]
        log10_phi2 = [self.params['gsmf_log10_phi_two_z0'], self.params['gsmf_log10_phi_two_z1'], self.params['gsmf_log10_phi_two_z2']]
        log10_mstar = [self.params['gsmf_log10_mstar_z0'], self.params['gsmf_log10_mstar_z1'], self.params['gsmf_log10_mstar_z2']]
        alpha1 = self.params['gsmf_alpha_one']
        alpha2 = self.params['gsmf_alpha_two']
        
        gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            
        return gsmf(10**mstar_log10 * MSOL, redshift)
    

    def get_shenf(self, redshift, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Retrieve the fit to the `Shen et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3252S/abstract>`_ bolometric luminosity function at a given redshift.
        
        Parameters
        -----------  
        redshift : float
            Redshift of the fit to be used 0.2-7.0 in steps of 0.2
        path_to_shen_fits : str, optional
            Path to the fits. Default is "/Users/cayenne/Documents/Research/quasarlf/qlffits/"

        Returns
        --------
        x : array
            The x-axis values, luminosity in log10(L/erg/s)
        y : array
            Fit to log phiL
        """
        dat = np.genfromtxt(path_to_shen_fits+"bolometric_fit_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']

    def get_shend(self, redshift, path_to_shen_data="/Users/cayenne/Documents/Research/quasarlf/qlfdata/"):
        """
        Retrieve the data from the `Shen et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3252S/abstract>`_  bolometric luminosity function at a given redshift.

        Parameters
        -----------
        redshift : float
            Redshift of the fit to be used 0.2-7.0 in steps of 0.2
        path_to_shen_fits : str, optional
            Path to the data. Default is "/Users/cayenne/Documents/Research/quasarlf/qlfdata/"

        Returns
        --------
        x : array
            The x-axis values, luminosity in log10(L/erg/s)
        y : array
            Data for log phiL
        """
        dat = np.genfromtxt(path_to_shen_data+"bolometric_data_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']
    
    def facfunc_zou(self, mbh_log10, redshift, facmin=0.00001):
        """
        Calculate AGN fraction as a function of stellar mass from `Zou et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...964..183Z/graphics>`_.
        Here stellar mass is inferred from black hole mass

        Parameters
        ----------
        mbh_log10 : array-like
            Log10 of black hole mass in solar masses
        redshift : float
            Redshift at which to evaluate the AGN fraction
        facmin : float, optional
            Minimum AGN fraction to return, default is 0.0, which is the value used in `Shen et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3252S/abstract>`_
        
        Returns
        -------
        phi_fa : array-like
            AGN fraction as a function of black hole mass and redshift
        """
        norm_fit = [-0.0348215, 0.77511731, -4.24506371]  # [-0.25916918189249905, 5.489176958654701, -31.25532992258093]
        slope_fit = [ 0.01273298, -0.28087742, 1.5361624 ]  # [0.2927610944131739, -6.537700910036786, 35.65876956759064]

        mamp_z = 10**self.params['mmb_mamp_log10'] * (1.0 + redshift)**self.params['mmb_zplaw_amp']
        mplaw_z = self.params['mmb_plaw'] * (1.0 + redshift)**self.params['mmb_zplaw_slope']

        mstar_log10 = (mbh_log10 - np.log10(mamp_z)) / mplaw_z + 11

        norm = norm_fit[0]*mstar_log10**2 + norm_fit[1]*mstar_log10 + norm_fit[2]
        norm[norm < 0.0] = 0.0
        slope = slope_fit[0]*mstar_log10**2 + slope_fit[1]*mstar_log10 + slope_fit[2]
        slope[slope > 0.0] = 0.0

        phi_fa = norm * (redshift) + slope

        phi_fa[phi_fa < facmin] = facmin

        return phi_fa
    
    def facfunc_quad(self, redshift, facmin=0.0):
        """
        Calculate AGN fraction as a function of redshift using a quadratic fit to data from `Zou et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...964..183Z/graphics>`_.

        Parameters
        ----------
        redshift : float
            Redshift at which to evaluate the AGN fraction
        facmin : float, optional
            Minimum AGN fraction to return. Default is 0.0
        
        Returns
        -------
        Factive : float
            AGN fraction as a function of redshift
        """
        a, b, c = -0.025714285714285707, 0.1685714285714286, -0.0806857142857146
        # a, b, c = -0.025714285714285707, 0.1685714285714286, -0.0406857142857146
        # a, b, c = -0.025714285714285707, 0.1685714285714286, -0.00806857142857146
        Factive = a * redshift**2 + b * redshift + c

        try:
            Factive[Factive < facmin] = facmin
        except TypeError:
            if Factive < facmin:
                Factive = facmin
        return Factive

    def facfunc_shan(self, mbh_log10, N0, alpha, beta, mbh_star):
        """
        Equation A1 from `Shankar et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013MNRAS.428..421S/abstract>`_
        for calculating the active fraction of black holes as a function of mass, where the active fraction is defined as the
        fraction of black holes that are actively accreting at a given time.

        Parameters
        ----------
        mbh_log10 : array-like
            Log10 of black hole mass in solar masses
        N0 : float
            Normalization
        alpha : float
            The low-mass slope
        beta : float
            The high-mass slope
        mbh_star : float
            The characteristic black hole mass at which the active fraction transitions from the low-mass slope to

        Returns
        -------
        Nactive : array-like
            The active fraction of black holes as a function of black hole mass
        """

        denom = (10**mbh_log10 / 10**mbh_star)**alpha + (10**mbh_log10 / 10**mbh_star)**beta

        Nactive = N0 / denom

        return Nactive
    
    def facfunc_cube(self, redshift):
        """
        Functional form assumed by `Wu et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260504776W/abstract>`_.
        Factive = 0.0004 * (1 + z)^3, here this is then multiplied by 10 since their AGN fraction is weighted.

        Parameters
        ----------
        redshift : float
            Redshift at which to evaluate the AGN fraction

        Returns
        -------
        Factive : float
            The active fraction of black holes as a function of redshift
        """
        Factive = 0.0004 * (1 + redshift)**3 * 10  # Multiply by 10 to approximate total AGN fraction
        try:
            Factive[Factive > 1.0] = 1.0
        except:
            if Factive > 1.0:
                Factive = 1.0
        return Factive
    
    def facfunc_interp(self, redshift, facmin=0.0):
        """
        Interpolated AGN fraction calculated by taking the ratio of the mass density between the BHMF predicted by comparing extrapolated methods in Shen et al. 2020 to the fiducial BHMF predicted by holodeck at many redshifts.
        
        .. note::
            This was calibrated using only BH masses > 10^8 Msun
        
        Parameters
        ----------
        redshift : float
            Redshift at which to evaluate the AGN fraction
        facmin : float, optional
            Minimum AGN fraction to return, default is 0.0  

        Returns
        -------
        Factive : float
            The active fraction of black holes as a function of redshift  
        """
        yvals = [0.01155760379692579, 0.024757591972556545, 0.0422131185738816, 0.06182450643481384, 0.08187823766345483, 0.10137354266892258, 0.11988088153114007, 0.13727233872774394, 0.15347464540720532, 0.16829037484606227, 0.18130133659343955, 0.18780050567575066, 0.19126556062586866, 0.19125173275475665, 0.18764478684847097, 0.1807696242254012, 0.17684420349963398, 0.17054284479086534, 0.16277088257305478, 0.15436150776267812, 0.14597480967203288, 0.1326862974740685, 0.12077011741805042, 0.11022740137039008, 0.10098672690611846, 0.09294512607316838, 0.08599544231733751, 0.08004474648745918, 0.07502851328716545]
        xvals = [0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5.0 , 5.2, 5.4, 5.6, 5.8]
        Factive = np.interp(redshift, xvals, yvals)
        try:
            Factive[Factive < facmin] = facmin
        except TypeError:
            if Factive < facmin:
                Factive = facmin
        return Factive

    def facfunc_interp_low(self, redshift, facmin=0.0):
        """
        Interpolated AGN fraction.
        
        Parameters
        ----------
        redshift : float
            Redshift at which to evaluate the AGN fraction
        facmin : float, optional
            Minimum AGN fraction to return, default is 0.0  

        Returns
        -------
        Factive : float
            The active fraction of black holes as a function of redshift  
        """
        # yvals = [0.00023609, 0.00031303, 0.00040196, 0.01062755, 0.02548049, 0.03880656, 0.0469578, 0.04854918, 0.06406479, 0.05828811, 0.05165821, 0.04513916, 0.03922779, 0.03457372, 0.02264437, 0.01873795, 0.01827168, 0.01202795, 0.01116053, 0.00605409, 0.00374844, 0.00210164, 0.00110246, 0.00053921, 0.00023606, 8.97096355e-05, 4.80775668e-05, 5.52178914e-05, 4.62499686e-05]
        # xvals = [0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5.0 , 5.2, 5.4, 5.6, 5.8]
        yvals = [9.76198246e-05, 3.28896622e-05, 3.81642940e-03, 1.88429600e-02, 4.10372524e-02, 6.00547615e-02, 7.40025222e-02, 7.32876803e-02, 6.78632547e-02, 6.06626936e-02, 5.32468483e-02, 4.62894149e-02, 4.00656023e-02, 3.49217484e-02, 2.45358509e-02, 2.00644951e-02, 1.61441979e-02, 1.25330360e-02, 9.05429422e-03, 6.12370468e-03, 3.75368935e-03, 2.09420264e-03, 1.09461657e-03, 5.32095814e-04, 2.29702206e-04, 8.38208013e-05, 4.33833980e-05, 5.34343953e-05, 4.58200689e-05, 3.36259944e-05, 2.35220237e-05, 1.61724891e-05, 1.10879335e-05, 7.70793256e-06]
        xvals = [0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5. , 5.2, 5.4, 5.6, 5.8, 6. , 6.2, 6.4, 6.6, 6.8]
        Factive = np.interp(redshift, xvals, yvals)
        try:
            Factive[Factive < facmin] = facmin
        except TypeError:
            if Factive < facmin:
                Factive = facmin
        return Factive
    
    def Prob_lam_Shen(self, loglambda_grid, mbh_log10, redshift, alpha=-0.6, lam1=1.5, mth=None):
        """
        Eddington ratio distribution function from `Shen et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3252S/abstract>`_ equation 30.

        Parameters
        ----------
        loglambda_grid : array
            Log10 of the Eddington ratio at which to evaluate the probability density function
        mbh_log10 : array
            Log10 of the black hole masses at which to evaluate the probability density function, shape should be (n_masses,)
        redshift : float
            Redshift at which to evaluate the probability density function
        alpha : float, optional
            The slope of the power law component of the distribution function. Default is -0.6.
        lam1 : float, optional
            The characteristic Eddington ratio at which the power law component turns over. Default is 1.5.
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.

        Returns
        -------
        prob : array
            The probability density function of log lambda at the given redshift, evaluated at the input log lambda values. Shape is (n_masses, n_loglambda_grid)
        """
        if mth is None:
            import numpy as mth
        loglam2 = mth.max([-1.9 + 0.45 * redshift, mth.log10(0.03)])
        sig = mth.max([1.03 - 0.15 * redshift, 0.6]) / mth.log(10)
        # F = 0.38  # Type-1 only, Compton thin fraction
        F = 0.99
        dlam = loglambda_grid[1] - loglambda_grid[0]
        dPt1 = (1 - F) * mth.power(10.0, loglambda_grid[mth.newaxis,:])**(1 + alpha) * mth.exp(-mth.power(10.0, loglambda_grid[mth.newaxis,:]) / lam1)
        Pt1 = mth.sum(dPt1) * dlam

        A = (1 - F) / Pt1 

        A = A * mth.ones(mbh_log10.shape[0])
        
        return (1 - F) * A[:,mth.newaxis] * (10**loglambda_grid[mth.newaxis,:])**(1 + alpha) * mth.exp(-10**loglambda_grid[mth.newaxis,:] / lam1) + F /\
        mth.sqrt(2 * mth.pi * sig**2) * mth.exp((-(loglambda_grid[mth.newaxis,:] - loglam2)**2 / (2 * sig**2)))

    def Prob_lam_Aird(self, loglambda_grid, mbh_log10, redshift, gamma1=-0.65, gamma2=-2.1, lambda_break=0.0, A=-3.15, beta=3.5, z0=0.6, mth=None):
        """
        Eddington ratio distribution function from `Aird et al. (2013) <https://iopscience.iop.org/article/10.1088/0004-637X/775/1/41/pdf>`_
        equation 1, has overal scatter of 0.38 dex. Individual reported uncertainties on input parameters are indicated below.

        Parameters
        ----------
        loglambda_grid : array
            Log10 of the Eddington ratio at which to evaluate the probability density function
        mbh_log10 : array
            Log10 of the black hole masses at which to evaluate the probability density function, shape should be (n_masses,)
        redshift : float
            Redshift at which to evaluate the probability density function
        gamma1 : float, optional
            Upper slope. Default is -0.65, +/- 0.04
        gamma2 : float, optional
            Lower slope. Default is -2.1 +0.3/-0.5, value of -1.65 matches the data in Model C, but -2.1 is from the abstract
        lambda_break : float, optional
            Knee. Default is 0.0
        A : float, optional
            Normalization. Default is -3.15 +/- 0.08
        beta : float, optional
            Parameter for redshift evolution. Default is 3.5 +/- 0.5
        z0 : float, optional
            Reference redshift. Default is 0.6
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.
        
        Returns
        -------
        prob : array
            The probability density function of lambda at the given redshift, evaluated at the input log lambda values. Shape is (n_masses, n_loglambda_grid)
        """
        if mth is None:
            import numpy as mth

        linear_lambda_grid = 10**loglambda_grid[mth.newaxis,:]
        beta = beta * mth.ones(mbh_log10.shape[0])

        p_low = (linear_lambda_grid)**gamma1

        p_hi = (linear_lambda_grid)**gamma2

        plam = 10**A * mth.where(loglambda_grid < lambda_break, p_low, p_hi) * ((1 + redshift)/(1 + z0))**beta[:,mth.newaxis]

        # dlam = loglambda_grid[1] - loglambda_grid[0]
        # norm = mth.sum(plam, axis=1, keepdims=True) * dlam
        norm = mth.sum(1/2 * (plam[:,1:] + plam[:,:-1]) * (loglambda_grid[1:] - loglambda_grid[:-1]), axis=1, keepdims=True)
        return plam / norm
    
    def Prob_lam_Ananna(self, linear_lambda_grid, dloglam, mbh_log10, delta1=0.38, eta_lambda=2.260, m=-0.885, b=6.671, mth=None):
        """
        Eddington ratio distribution function from `Ananna et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022ApJS..261....9A/abstract>`_ Equation 11 and Table 4.

        Parameters
        ----------
        linear_lambda_grid : array
            The Eddington ratio at which to evaluate the probability density function
        redshift : float
            Redshift at which to evaluate the probability density function
        volume : float
            Comoving volume at the given redshift
        mbh_log10 : array
            Log10 of the black hole masses at which to evaluate the probability density function, shape should be (n_masses,)
        zeta_star : float
            Normalization of the distribution function, this version is normalized to be a probability function so this value does not affect the output. Default is 10**-3.64
        delta1 : float
            Power law slope at low Eddington ratios. Default is 0.38
        eta_lambda : float
            Parameter controlling the steepness of the exponential cutoff at high Eddington ratios. Default is 2.260
        m : float
            Slope of the mass dependence of the characteristic Eddington ratio. Default is -0.885
        b : float
            Intercept of the mass dependence of the characteristic Eddington ratio. Default is 6.671
        mth : module, optional
            Module to use for mathematical functions. Default is None.

        Returns
        -------
        prob : array
            The probability density function of lambda at the given redshift, evaluated at the input log lambda values. Shape is (n_masses, n_loglambda_grid)
        """
        if mth is None:
            import numpy as mth

        lambda_star = 10**(mbh_log10 * m + b)  # Default single value is 10**-1.338

        ratio = linear_lambda_grid[mth.newaxis,:] / lambda_star[:,mth.newaxis]

        plam = 1 / (ratio**delta1 * (1 + ratio**eta_lambda))

        # plam = mth.where(linear_lambda_grid < 0.0001, 1e-100, plam)
        norm = mth.sum(plam, axis=1, keepdims=True) * dloglam
        # norm = mth.sum(1/2 * (plam[:,1:] + plam[:,:-1]) * (linear_lambda_grid[1:] - linear_lambda_grid[:-1]), axis=1, keepdims=True)
        return plam / norm
    
    def Prob_lam_Three(self, linear_lambda_grid, mbh_log10, redshift, alpha=1, beta=-0.65, gamma=-2.1, lam1=10**-4, lam2=10**0, mth=None):
        """
        Eddington ratio distribution function from `Aird et al. (2013) <https://iopscience.iop.org/article/10.1088/0004-637X/775/1/41/pdf>`_
        equation 1, has overal scatter of 0.38 dex. Individual reported uncertainties on input parameters are indicated below.

        Parameters
        ----------
        loglambda_grid : array
            Log10 of the Eddington ratio at which to evaluate the probability density function
        mbh_log10 : array
            Log10 of the black hole masses at which to evaluate the probability density function, shape should be (n_masses,)
        redshift : float
            Redshift at which to evaluate the probability density function
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.
        
        Returns
        -------
        prob : array
            The probability density function of lambda at the given redshift, evaluated at the input log lambda values. Shape is (n_masses, n_loglambda_grid)
        """
        if mth is None:
            import numpy as mth

        beta = beta * mth.ones(mbh_log10.shape[0])

        p_low = lam1**(beta-alpha) * (linear_lambda_grid[mth.newaxis,:])**alpha

        p_mid = (linear_lambda_grid[mth.newaxis,:])**beta[:,mth.newaxis]

        p_hi = lam2**(beta-gamma) * linear_lambda_grid[mth.newaxis,:]**gamma

        plam_temp = mth.where(linear_lambda_grid < lam1, p_low, p_mid)
        plam = mth.where(linear_lambda_grid < lam2, plam_temp, p_hi)

        norm = mth.sum(1/2 * (plam[:,1:] + plam[:,:-1]) * (linear_lambda_grid[1:] - linear_lambda_grid[:-1]), axis=1, keepdims=True)
        return plam / norm

    def Prob_lam_Inactive(self, loglambda_grid, loglam_norm=-10, sig=1, mth=None):
        """
        Eddington ratio distribution function for inactive black holes.

        Parameters
        ----------
        loglambda_grid : array
            Log10 of the Eddington ratio at which to evaluate the probability density function
        loglam_norm : float, optional
            Median log10 value of Eddington fraction for the inactive black holes. Default is -10
        sig : float, optional
            Scatter on the distribution. Default is 1
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.

        Returns
        -------
        prob : array
            The probability density function of log lambda at the given redshift, evaluated at the input log lambda values. Shape is (n_masses, n_loglambda_grid)
        """
        if mth is None:
            import numpy as mth

        if loglam_norm is None:
            loglam_norm = -10
        if sig is None:
            sig = 1
        
        lln = mth.ones(loglambda_grid.shape[0]) * loglam_norm
        plam =  1 / mth.sqrt(2 * mth.pi * sig**2) * mth.exp((-(loglambda_grid[mth.newaxis,:] - lln[:,mth.newaxis])**2 / (2 * sig**2)))
        
        norm = mth.sum(1/2 * (plam[:,1:] + plam[:,:-1]) * (loglambda_grid[1:] - loglambda_grid[:-1]), axis=1, keepdims=True)
        return plam / norm
            
    def Prob_lam_Fractional(self, loglambda_grid, Factive, Ploglam_active, loglam_norm=None, sig=None, mth=None):
        """
        Probability of loglambda with two peaks. One peak for active black holes and one peak for inactive black holes.
        The relative contributions of each peak is determined by the active fraction.

        Parameters
        ----------
        loglambda_grid : array
            Log10 of the Eddington ratio at which to evaluate the probability density function
        Factive : float or array
            The active fraction(s)
        Ploglam_active : array
            The probabilitiy distribution function of active black holes
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.

        Returns
        -------
        Plam_tot : array
            Normalized probability density in dlog10(lambda). Shape is (n_masses, n_loglambda_grid)
        """

        if mth is None:
            import numpy as mth

        F = mth.ones(Ploglam_active.shape[0])*Factive

        Ploglam_inactive = self.Prob_lam_Inactive(loglambda_grid, mth=mth, loglam_norm=loglam_norm, sig=sig)

        Plam_tot = Ploglam_inactive * (1 - F[:,None]) + Ploglam_active * F[:,None]

        return Plam_tot

    def PhiM_to_PhiL_erdf(self, mbh_log10, phiM, redshift, logL_grid, lambda_grid, Pfunc='Shen', Fractional=False, facfunc='Interp', mth=None, loglam_norm=None, sig=None, **kwargs):
        """
        Convert a BH mass function into an AGN luminosity function.

        Parameters
        ----------
        mbh_log10 : array
            log10(M_BH / Msun)
        phiM : array
            Phi_BH(logM)
        logL_grid : array
            log10(Lbol / erg s^-1)
        lambda_grid : array
            Eddington ratio grid, may be in log10 or linear space depending on the Pfunc used
        Pfunc : str or float
            Function to use for the ERDF, options are 'Shen' and 'Aird'. Default is 'Shen'.
        Fractional : bool, optional
            Flag for how to use the active fraction. Default is False.
        facfunc : str or int, optional
            Which functional form to use for calculating AGN fraction, options are 'Zou', 'Quad', 'Cube' 'Interp', and 'Interp_low'. Default is 'Interp'.
        mth : module, optional
            Module to use for mathematical functions. Default is None which sets mth = numpy.
        kwargs : optional
            Input parameters for the probability density function of log lambda

        Returns
        -------
        phiL : array
            The number density per unit log luminosity for the AGN luminosity function
        """

        if mth is None:
            import numpy as mth

        #################################

        # linear_lambda_grid = 10**loglambda_grid
        dloglam = lambda_grid[1] - lambda_grid[0]
        
        phiL = mth.zeros_like(logL_grid)

        phiL_list = []

        for logL in logL_grid:
            logM_edd = (logL - lambda_grid - mth.log10(C_edd))

            phiM_interp = mth.interp(logM_edd, mbh_log10, phiM, right=0.0)

            if facfunc == 'Zou':
                Factive = self.facfunc_zou(mbh_log10=logM_edd, redshift=redshift)

            elif facfunc == 'Quad':
                Factive = self.facfunc_quad(redshift=redshift)
            
            elif facfunc == 'Cube':
                Factive = self.facfunc_cube(redshift=redshift)

            elif facfunc == 'Interp':
                Factive = self.facfunc_interp(redshift=redshift)

            elif facfunc == 'Interp_low':
                Factive = self.facfunc_interp_low(redshift=redshift)
            
            elif type(facfunc) != str:
                Factive = facfunc

            if Pfunc == 'Shen':
                Ploglam = self.Prob_lam_Shen(loglambda_grid=lambda_grid, mbh_log10=logM_edd, redshift=redshift, mth=mth, **kwargs)
    
            elif Pfunc == 'Aird':
                Ploglam = self.Prob_lam_Aird(loglambda_grid=lambda_grid, mbh_log10=logM_edd, redshift=redshift, mth=mth, **kwargs)

            elif Pfunc == 'Ananna':
                Ploglam = self.Prob_lam_Ananna(linear_lambda_grid=lambda_grid, dloglam=dloglam, mbh_log10=logM_edd, mth=mth, **kwargs)
            
            elif Pfunc == 'Three':
                Ploglam = self.Prob_lam_Three(linear_lambda_grid=lambda_grid, mbh_log10=logM_edd, redshift=redshift, mth=mth, **kwargs)

            if Fractional == True:
                if facfunc == 'Zou':
                    raise TypeError("Cannot use fractional Ploglam with Zou Factive yet. Please choose another Factive function.")
                Ploglam = self.Prob_lam_Fractional(lambda_grid, Factive=Factive, Ploglam_active=Ploglam, mth=mth, loglam_norm=loglam_norm, sig=sig)
                Factive = 1  # This avoids double counting Factive in the integral below

            y = phiM_interp * Ploglam * Factive

            dlam = lambda_grid[1:] - lambda_grid[:-1]

            integrand = (mth.sum(0.5 * dlam[:, None] * (y[1:] + y[:-1])) / logM_edd.shape[0])
            # integrand = (mth.sum(0.5 * dlam[None, :] * (y[:, :-1] + y[:, 1:])) / logM_edd.shape[0])

            phiL_list.append(integrand)

        phiL = mth.stack(phiL_list)
        return phiL