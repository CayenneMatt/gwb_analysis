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

import copy

"""

"""

class Model_Info(object):
    """
    Read the model data and store relevant features as attributes.
    """
    # Attributes
    def __init__(self, path, file, model_name, color, line_style, threshold, evolving=False, stdev=None, nfreq=5, param_space_name = 'PS_Astro_Strong_All'):

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
        self.nfreq = nfreq  # Number of frequency bits to fit to, default is 5
        self.idcs = None  # Index of evolution parameter

        id = np.where((self.param_names == 'mmb_zplaw') | (self.param_names == 'mmb_zplaw_amp') | (self.param_names == 'mmb_zplaw_slope') | (self.param_names == 'mmb_zplaw_scatter'))[0]
        if len(id) > 0:
            self.idcs = id
        
        # Dictionary of priors to be modified by get_posteriors(), note that LM* models have different priors, but currently sample every parameter.
        space_class = librarian.param_spaces_dict[self.param_space_name]
        self.fiducial_values = copy.deepcopy(space_class.DEFAULTS)
        self.posteriors = copy.deepcopy(space_class.DEFAULTS)
        
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

        space_class = librarian.param_spaces_dict[self.param_space_name]
        space = space_class(nsamples=int(1e4)) # Draw samples from the parameter space with LHC

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
            self.posteriors[n] = median

        return self
    
    def corner_plot(self, nbins=20, cmap='Blues'):

        npars = len(self.param_names)
        nsamp, npars = self.params.shape

        gwb_med = np.median(self.gwb, axis=-1)

        valid = np.any(gwb_med > 0, axis=1)
        like = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sort_idx = np.argsort(like)

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

            ww = np.exp(like[sort_idx[valid]])
            if ii == jj:
                ax.hist(xx, histtype='step', bins=edges[jj], alpha=0.5, density=True, color='k', ls='--')
                ax.hist(xx, histtype='step', bins=edges[jj], weights=ww, alpha=0.5, density=True, color=colorsc[-1], lw=1.5)

                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                continue

            yy = self.params[:, ii][sort_idx[valid]]
            bins = (edges[jj], edges[ii])
            kale.contour((xx, yy), edges=bins, weights=ww, ax=ax, pad=0, smooth=1.5);
    
    def bhmf(self, mass, redz, fiducial=False):
        """
        Produces the black hole mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Arguments:
        mass: tuple (min, max, npoints) in log10(Mbh/Msol) to be used as arguments in np.linspace()
        redz: redshift
        -----------
        Returns: (bhmf, None) where bhmf is the black hole mass function at the given redshift
        """
        masses = np.linspace(mass[0], mass[1], mass[2])

        if not fiducial:
            log10_phi1 = [self.posteriors['gsmf_log10_phi_one_z0'], self.posteriors['gsmf_log10_phi_one_z1'], self.posteriors['gsmf_log10_phi_one_z2']]
            log10_phi2 = [self.posteriors['gsmf_log10_phi_two_z0'], self.posteriors['gsmf_log10_phi_two_z1'], self.posteriors['gsmf_log10_phi_two_z2']]
            log10_mstar = [self.posteriors['gsmf_log10_mstar_z0'], self.posteriors['gsmf_log10_mstar_z1'], self.posteriors['gsmf_log10_mstar_z2']]
            alpha1 = self.posteriors['gsmf_alpha_one']
            alpha2 = self.posteriors['gsmf_alpha_two']
            
            gsmf = sams.GSMF_Double_Schechter(log10_phi1, log10_phi2, log10_mstar, alpha1, alpha2)
            mmb = holo.host_relations.MMBulge_Redshift_KH2013(mamp_log10 = self.posteriors['mmb_mamp_log10'],
                                                                mplaw = self.posteriors['mmb_plaw'],
                                                                zplaw_amp=self.posteriors['mmb_zplaw_amp'],
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
        
        return masses, gsmf.mbh_mass_func_conv(10**masses * MSOL, redz, mmbulge=mmb, scatter=True), None
    
    def add_spectrum(self, ax, errorbars=False, label=None):
        if not label:
            label = self.model_name
        vals = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sim_idx = np.argmax(vals)
        gwb = np.median(self.gwb, axis=2)
        ax.plot(np.log10(self.freqs), gwb[sim_idx], lw=3, ls=self.line_style, c=self.color, label=label)
        if errorbars:
            valid = np.where((vals >= (np.nanmax(vals)- self.threshold)))[0]
            up = np.max(gwb[valid], axis=0)
            dn = np.min(gwb[valid], axis=0)
            ax.fill_between(np.log10(self.freqs), up, dn, color=self.color, alpha=0.25)


