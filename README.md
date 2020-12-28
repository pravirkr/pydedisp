# pydedisp

`pydedisp` is a GPU/CPU De-dedispersion library for python.

[![GitHub CI](https://github.com/pravirkr/pydedisp/workflows/Build/badge.svg)](https://github.com/pravirkr/pydedisp/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7993c188ab344bb9af9cf8de0236615a)](https://www.codacy.com/gh/pravirkr/pydedisp/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pravirkr/pydedisp&amp;utm_campaign=Badge_Grade)

## Introduction

Based on Ben Barsdell's original [GPU De-dedispersion library](https://github.com/ajameson/dedisp), pydedisp uses
a GPU to perform the computationally intensive task of computing the incoherent dedispersion transform, a frequent
operation in signal processing and time-domain radio astronomy.

dedisp provides a simple C interface to computing dedispersion transforms using a GPU. The interface is modelled on
that of the well-known [FFTW](http://www.fftw.org/) library, and uses an object-oriented approach. The user first
creates a dedispersion _plan_, and then calls functions to modify or execute that plan. A full list and description
of the functions provided by the library can be viewed in the API documentation below.

For a detailed discussion of how the library implements dedispersion on the GPU,
see [Barsdell et al. 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.422..379B).
If you use the library in your work, please consider citing this paper.

## Features

* High performance: up to **10x speed-up** over efficient quad-core CPU implementation
* **Pure C interface** allows easy integration into existing C/C++/Fortran/Python etc. codes
* Accepts input time series sampled with **1, 2, 4, 8, 16 or 32 bits per sample**
* Can produce dedispersed time series sampled with **8, 16 or 32 bits per sample**
* **DM trials can be generated** by the library **or supplied by the user**
* Accepts a channel **'killmask'** for skipping bad channels
* **Adaptive time resolution** (aka _time-scrunching_, _binning_) for further speed gains
* Input and output data can be passed **from the host or directly from the GPU**
* Extended **'advanced' and 'guru' interfaces** allow arbitrary data strides and DM gulping
* Optional **C++ wrapper** for convenient object-oriented syntax

## Dependencies

The library requires NVIDIA's [CUDA](https://developer.nvidia.com/cuda-zone) in order to access the GPU.
This also imposes the constraint that the target hardware must be an NVIDIA GPU.
To compile the library you must have the NVIDIA CUDA C compiler _nvcc_ in your path.

* [CUDA 10.0+](https://developer.nvidia.com/cuda-toolkit-archive)
* [CMake 3.0+](https://cmake.org/download/)
* [Doxygen](http://www.doxygen.nl/download.html)
* [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html)
* [Breathe](https://pypi.org/project/breathe/)
* [sphinx_rtd_theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)

## Installation

1. Update Makefile.inc with your CUDA path, Install Dir and GPU architecture. e.g.
    * CUDA_PATH ?= /usr/local/cuda-8.0.61
    * INSTALL_DIR = $(HOME)/opt/dedisp
    * GPU_ARCH = sm_60
2. mkdir build && cd build && cmake ..
3. make && make install

This will build a shared object library named libdedisp.so which is a prerequisite for Heimdall.
The dedisp header files will be installed into INSTALL_DIR/include and the library into INSTALL_DIR/lib.

## Debugging

The following make flag can be used to build the library in a
debug mode, which will print the location and description of errors as
they happen:

`$ make DEDISP_DEBUG=1`
