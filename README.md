# pydedisp

`pydedisp` is a GPU/CPU De-dedispersion library for python.

[![GitHub CI](https://github.com/pravirkr/pydedisp/workflows/Build/badge.svg)](https://github.com/pravirkr/pydedisp/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7993c188ab344bb9af9cf8de0236615a)](https://www.codacy.com/gh/pravirkr/pydedisp/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pravirkr/pydedisp&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/pravirkr/pydedisp/branch/master/graph/badge.svg)](https://codecov.io/gh/pravirkr/pydedisp)

## Introduction

Based on Ben Barsdell's original [GPU De-dedispersion library](https://github.com/ajameson/dedisp), pydedisp uses
a GPU to perform the computationally intensive task of computing the incoherent dedispersion transform, a frequent
operation in signal processing and time-domain radio astronomy.

For a detailed discussion of how the library implements dedispersion on the GPU,
see [Barsdell et al. 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.422..379B).
If you use the library in your work, please consider citing this paper.

## Dependencies

The library requires NVIDIA's [CUDA](https://developer.nvidia.com/cuda-zone) in order to access the GPU.

* [CUDA 10.0+](https://developer.nvidia.com/cuda-toolkit-archive)
* [CMake 3.15+](https://cmake.org/download/)

## Installation (C++ interface)

1. Update CMakeLists.txt with your CUDA path and GPU architecture.
2. mkdir build && cd build && cmake ..
3. make && make install

This will build a shared object library named libdedisp.so which is a prerequisite for Heimdall.
The dedisp header files will be installed into INSTALL_DIR/include and the library into INSTALL_DIR/lib.

### with pip

Once you have all the requirements installed, you can install this via pip:

```bash
pip install git+https://github.com/pravirkr/pydedisp
```

Or, download / clone this repository, and then run

```bash
python -m pip install .
```
