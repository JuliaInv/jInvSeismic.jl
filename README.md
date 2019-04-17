

[![Build Status](https://travis-ci.org/JuliaInv/jInvSeismic.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/jInvSeismic.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/jInvSeismic.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/jInvSeismic.jl?branch=master)

# jInvSeismic.jl - soon to be a collection of seismic inversion related packages 

# jInvSeismic.BasicFWI - a lightweight basic Full Waveform Inversion in Julia.
 

# Requirements

This package is inteded to use with Julia versions 0.7.

This package is an add-on for [`jInv`](https://github.com/JuliaInv/jInv.jl), which needs to be installed. This is a basic FWI package, mostly used for teaching.

# Installation

In julia type:

``` 
Pkg.clone("https://github.com/JuliaInv/jInv.jl","jInv")
Pkg.clone("https://github.com/JuliaInv/jInvSeismic.jl","jInvSeismic")
Pkg.test("jInvSeismic")
```
