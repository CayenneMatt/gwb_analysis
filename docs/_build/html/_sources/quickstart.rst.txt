Quickstart
==========

The primary module, analyze_model, has one class, Model_Info with many attributes and methods. This may get split into multiple modules eventually.

This class enables posterior calculation and plotting. Here is an example of how to read in some data and caclulate posteriors:


.. code-block:: python
   
   from gwb_analysis.gwb_analysis.analyze_model import Model_Info
   model1 = Model_Info(path='path/to/file/', file='model_file.npz', model_name='Model1', color='steelblue', line_style='--', threshold=2.5).get_posteriors()

