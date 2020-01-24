using Distributed
if nworkers()==1
	addprocs(1)
end
using SparseArrays
using LinearAlgebra
using jInv.Mesh
using jInv.LinearSolvers
using jInv.ForwardShare
using jInv.InverseSolve
using jInv.Utils
using Printf
using Helmholtz
using jInvSeismic.FWI
using jInvSeismic.Utils
using jInvSeismic.EikonalInv
using Multigrid
using Statistics
using MAT
using DelimitedFiles
using Test

include("testGetData.jl")
include("testSensitivity.jl")
include("testInversion.jl")
#include("testTimeDomain.jl")