CommonRoad-Geometric
=============

*commonroad-geometric* is a Python framework that facilitates deep-learning based research projects in the autonomous driving domain, e.g. related to behavior planning and state representation learning.
At its core, it provides a standardized interface for heterogeneous graph representations of traffic scenes using the PyTorch Geometric framework.
The package aims to serve as a flexible framework that, without putting restrictions on potential research directions, minimizes the time spent on implementing boilerplate code. Through its object-oriented design with highly flexible and extendable class interfaces, it is meant to be imported via pip install and utilized in a plug-and-play manner.

The software is written in Python and tested on Linux for the Python 3.8, 3.9, 3.10, and 3.11.

Documentation
=============

The full documentation of the API and introducing examples can be found under `commonroad.in.tum.de <https://commonroad-geometric.readthedocs.io/en/latest/>`__.

Installation
============

commonroad-geometric can be installed with::

	pip install commonroad-geometric

Alternatively, clone from our gitlab repository::

	git clone https://github.com/CommonRoad/commonroad-geometric

and add the folder commonroad-geometric to your Python environment.

The installation script `create-dev-environment.sh <https://github.com/CommonRoad/commonroad-geometric/-/blob/develop/scripts/create-dev-environment.sh>`_ installs the commonroad-geometric package and all its dependencies into a conda environment.
Execute the script inside the directory which you want to use for your development environment.

Getting Started
===============

A tutorial on the main functionalities of the project is :ref:`available here<getting_started>`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user/index.rst
   api/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contact information
===================

:Website: `http://commonroad.in.tum.de <https://commonroad.in.tum.de/>`_
:Email: `commonroad@lists.lrz.de <commonroad@lists.lrz.de>`_
