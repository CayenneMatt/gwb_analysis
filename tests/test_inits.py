# This test will import important modules and libraries.
def test_imports():
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
