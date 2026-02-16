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

"""
TODO:
* Add Unit tests to make sure this works beyond visual comparison
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
        self.space_class = librarian.param_spaces_dict[self.param_space_name]
        self.fiducial_values = copy.deepcopy(self.space_class.DEFAULTS)
        self.posteriors = copy.deepcopy(self.space_class.DEFAULTS)
        # self.posteriors_err = {k: np.abs(self.space_class.DEFAULTS[k]*0.1) for k in self.space_class.DEFAULTS}

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
            stdev = rv.std()
            self.posteriors[n] = median
            self.posteriors_err[n] = stdev

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
    
    def add_spectrum(self, ax, lw=3, errorbars=False, label=None):
        if not label:
            label = self.model_name
        vals = np.sum(self.ln_like[:, :self.nfreq], axis=-1)
        sim_idx = np.argmax(vals)
        gwb = np.median(self.gwb, axis=2)
        ax.plot(np.log10(self.freqs), gwb[sim_idx], lw=lw, ls=self.line_style, c=self.color, label=label)
        if errorbars:
            valid = np.where((vals >= (np.nanmax(vals)- self.threshold)))[0]
            print(len(valid) / len(vals))
            up = np.max(gwb[valid], axis=0)
            dn = np.min(gwb[valid], axis=0)
            ax.fill_between(np.log10(self.freqs), up, dn, color=self.color, alpha=0.25)

        
    def bhmf(self, mass, redz, fiducial=False):
        """
        Produces the black hole mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Arguments:
        mass: tuple (min, max, npoints) in log10(Mbh/Msol) to be used as arguments in np.linspace()
        redz: redshift
        -----------
        Returns: (masses, bhmf, None) where bhmf is the black hole mass function at the given redshift
        """
        masses = np.linspace(mass[0], mass[1], mass[2])

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
        
        return masses, gsmf.mbh_mass_func_conv(10**masses * MSOL, redz, mmbulge=mmb, scatter=True)
    
    def bhmf_err(self, mass, redz, ndraws=100):
        """
        Produces the black hole mass function at a given redshift. Calculated using holodeck by connvolving a double Schechter GSMF with a MMBulge relation

        Arguments:
        mass: tuple (min, max, npoints) in log10(Mbh/Msol) to be used as arguments in np.linspace()
        redz: redshift
        -----------
        Returns: (masses, bhmf_median, bhmf_upper_bound, bhmf_lower_bound) where bhmf is the black hole mass function at the given redshift

        ------------
        TODO:
        * Add fiducial functionality
        """
        masses = np.linspace(mass[0], mass[1], mass[2])
        
        self.bhmf_dict = copy.deepcopy(self.space_class.DEFAULTS)

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
            
            phis.append(gsmf.mbh_mass_func_conv(10**masses * MSOL, redz, mmbulge=mmb, scatter=True))

        phi_50, phi_84, phi_16 = np.nanpercentile(phis, [50, 84, 16], axis = 0)
        
        return masses, phi_50, phi_84, phi_16
    

    def get_shenf(self, redshift, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Retrieve the fit to the Shen+2020 bolometric luminosity function at a given redshift.
        
        Arguments:     
        redshift: redshift of the fit to be used 0.2-7.0 in steps of 0.2
        path_to_shen_fits = '/Users/cayenne/Documents/Research/quasarlf/qlffits/': Path to the Shen+2020 fits for the bolometric LF at different redshifts

        --------------

        Returns: x (logL), y (log phiL)

        """
        dat = np.genfromtxt(path_to_shen_fits+"bolometric_fit_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']

    def get_shend(self, redshift, path_to_shen_data="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Retrieve the data from the Shen+2020 bolometric luminosity function at a given redshift.

        Arguments:
        redshift: redshift of the data to be used 0.2-7.0 in steps of 0.2
        path_to_shen_data = '/Users/cayenne/Documents/Research/quasarlf/qlffits/': Path to the Shen+2020 data for the bolometric LF at different redshifts

        --------------

        Returns: x (logL), y (log phiL)
        """
        dat = np.genfromtxt(path_to_shen_data+"bolometric_data_"+str(redshift)+".txt", dtype=None, encoding=None, names=True)
        return dat['x'], dat['y']

    def calulate_radiative_efficiency(self, zval, step, mass=[5, 13, 100], fiducial=False, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Calculate the radiative efficiency implied by the model at a given redshift by comparing the change in the black hole mass function between two redshifts to the
        luminosity function at the average redshift. This is a rough calculation that assumes that the change in the BHMF and LF between the two redshifts is solely due to accretion
        and does not consider mergers or other processes that may contribute to the growth of black holes.

        Arguments:
        z1: lower redshift
        z2: higher redshift
        fiducial: whether to use the fiducial values of the model parameters instead of the posteriors
        path_to_shen_fits = '/Users/cayenne/Documents/Research/quasarlf/qlffits/': path to the Shen+2020 fits for the bolometric LF at different redshifts

        --------------

        Returns: radiative efficiency between the two redshifts

        --------------

        Issues:

        * Incorporate AGN Fraction
        * Radiative efficiency may be mass and redshift dependent, not currently implemented
        * Need to make sure that it is always integrating over roughly similar mass - luminosity ranges
        * Integration only cosiders two bins and nothing in between, but should be fine for order of magnitude calculation

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

        return erad.decompose()


    def fit_AGN_Luminosity(self, zval, step, mass=[5, 13, 100], fiducial=False, path_to_shen_fits="/Users/cayenne/Documents/Research/quasarlf/qlffits/"):
        """
        Calculate the AGN luminosity implied by the model at a given redshift, assume erad = f_obsc = f_acc = 1 so that they can be fit for.
        """
        # Get BHMF at each redshift

        z1 = zval
        z2 = zval + step

        dt = cosmo.lookback_time(z2) - cosmo.lookback_time(z1)
        volume = cosmo.comoving_volume(z2) - cosmo.comoving_volume(z1)

        # Mass Function
        masses, bhmf1 = self.bhmf(mass, redz=z1, fiducial=fiducial)
        masses, bhmf2 = self.bhmf(mass, redz=z2, fiducial=fiducial)
        
        mdot = (trapz((bhmf1 - bhmf2) * 10**masses, masses) * u.Msun / dt)

        # Luminosity Calculation
        Lum = (mdot * c.c**2).decompose().to(u.erg / u.s).value

        return Lum

