Quickstart
==========

The primary module, analyze_model, has one class, Model_Info with many attributes and methods. This may get split into multiple modules eventually.

This class enables posterior calculation and plotting. The examples below are written to use the example data hosted on `Dropbox <https://www.dropbox.com/scl/fo/w4bimbjtu0wgl3tsyekvj/ABygWjCa5DbX1z_n3AFwDiQ?rlkey=30pxb09xdkkiteag0twheyx85&st=pf2w39rz&dl=0>`_. These files are the "Le03ne" and "Le03ev" models from `Matt et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026ApJ...997..188M/abstract>`_.

Here is an example of how to read in some data and caclulate posteriors.


.. code-block:: python
   
   from gwb_analysis.gwb_analysis.analyze_model import Model_Info
   model1 = Model_Info(path='path/to/file/', file='model_file.npz', model_name='Model1',
   color='steelblue', line_style='--', threshold=0.5).get_posteriors()

The object "model1" has a "params" attribute which is a dictionary of parameter values. The entries in this dictionary will be the default values until "get_posteriors()" is run, after which the entries will be the medians of the posterior distributions.

Once the model object is created, you can look at the maximum likelihood spectrum.
   
.. code-block:: python

   import matplotlib.pyplot as plt
   from gwb_analysis.gwb_analysis.analyze_model import Model_Info
   model1 = Model_Info(path='path/to/file/', file='model_file.npz', model_name='Model1',
   color='steelblue', line_style='--', threshold=0.5).get_posteriors()

   fig, ax = plt.subplots(1, 1, figsize=(10, 7))
   model1.add_spectrum(ax=ax, errorbars=True)
   ax.set_yscale('log')
   ax.set_xlim(-8.8, -7.55)
   ax.set_ylim(1e-16, 3e-14);
   ax.set_ylabel(r"Strain amplitude, $h_c$")
   ax.set_xlabel("log GW Frequency [Hz]");
   ax.legend(loc='lower left', frameon=False);


Or create a corner plot for all the posterior parameters

.. caution::
   The get_posteriors() method must be run before making a corner plot, otherwise the posteriors will remain identical to the priors.

.. code-block:: python

   from gwb_analysis.gwb_analysis.analyze_model import Model_Info
   model1 = Model_Info(path='path/to/file/', file='model_file.npz', model_name='Model1',
   color='steelblue', line_style='--', threshold=0.5).get_posteriors()

   model1.corner_plot()

Or a black hole mass function.

.. code-block:: python
   
   import matplotlib.pyplot as plt
   import numpy as np
   from gwb_analysis.gwb_analysis.analyze_model import Model_Info
   model1 = Model_Info(path='path/to/file/', file='model_file.npz', model_name='Model1',
   color='steelblue', line_style='--', threshold=0.5).get_posteriors()

   mbh_log10 = np.linspace(5, 13, 100)
   for z in (0, 1, 2):

       ndens = model1.bhmf(mbh_log10, redz=z)
       plt.plot(mbh_log10, ndens, color=model1.color, ls=model1.line_style,
       label=model1.model_name)

   plt.yscale('log')
   plt.xlim(7, 13)
   plt.ylim(1e-8, 1e-1)
   plt.legend(loc='upper right', frameon=False);
