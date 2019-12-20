

[![Build Status](https://travis-ci.org/JuliaInv/jInvSeismic.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/jInvSeismic.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/jInvSeismic.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/jInvSeismic.jl?branch=master)

# jInvSeismic.jl - soon to be a collection of seismic inversion related packages 

# Overview

jInvSeismic consists of submodules:

1. `jInvSeismic.FWI` - A Julia package for solving the Full Waveform Inversion on a regular rectangular mesh. For forward modelling and sensitivities it uses either a direct solver or a block Shifted Laplacian Multigrid preconditioner with BiCGSTAB.
2. `jInvSeismic.EikonalInv` - A Julia package for solving the inverse eikonal equation on a regular rectangular mesh.
For forward modelling and sensitivities it uses the fast marching algorithm for the factored eikonal equation.
3. `jInvSeismic.BasicFWI` - a lightweight basic Full Waveform Inversion in Julia.
4. `jInvSeismic.Utils` - utility functions for seismic inversions.

# Literature
The EikonalInv package is based on the following paper (please cite if you are using the package):

Eran Treister and Eldad Haber, A fast marching algorithm for the factored eikonal equation, Journal of Computational Physics, 324, 210-225, 2016.

The EikonalInv and FWI packages are used in the following papers in joint inversions:

Lars Ruthotto, Eran Treister and Eldad Haber, jInv--a flexible Julia package for PDE parameter estimation, SIAM Journal on Scientific Computing, 39 (5), S702-S722, 2017. 

Eran Treister and Eldad Haber, Full waveform inversion guided by travel time tomography, SIAM Journal on Scientific Computing, 39 (5), S587-S609, 2017.

# Requirements

This package is inteded to use with Julia versions 1.2.

This package is an add-on for [`jInv`](https://github.com/JuliaInv/jInv.jl), which needs to be installed. 

# Installation

In julia type:

``` 
Pkg.clone("https://github.com/JuliaInv/jInv.jl","jInv")
Pkg.clone("https://github.com/JuliaInv/FactoredEikonalFastMarching.jl","FactoredEikonalFastMarching")
Pkg.clone("https://github.com/JuliaInv/jInvSeismic.jl","jInvSeismic")
Pkg.test("jInvSeismic")
```

# Examples

Under "examples/TravelTimeTomography/SEGTravelTimeInversionExample.jl" you can find the 2D experiment that was shown in the paper above. 




