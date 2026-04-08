Installation
============

This module is not yet pip installable. If you clone the main `GitHub Page <https://github.com/CayenneMatt/gwb_analysis>`_ you can then locally pip install from within the Git repository:

.. code-block:: console

   pip install -e .

There is no requirements.txt file, but this library is dependent on `holodeck <https://github.com/nanograv/holodeck>`_. Make sure you have holodeck installed and functioning:

.. code-block:: console

   conda create --name holo311 python=3.11
   conda activate holo311
   git clone https://github.com/nanograv/holodeck.git
   cd holodeck

   conda install -c conda-forge numba llvmlite numpy scipy matplotlib -y
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   python setup.py build_ext -i
   python setup.py develop
   conda install -c conda-forge mpi4py openmpi

Optional:

.. code-block:: console

   pip install chainconsumer==0.33.0 scipy==1.10 la_forge
