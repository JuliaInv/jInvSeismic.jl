using Test
using Distributed
if nworkers() == 1
	addprocs(2);
elseif nworkers() < 2
	addprocs(2 - nworkers());
end

using SparseArrays
using LinearAlgebra
@everywhere using jInv.Mesh
@everywhere using jInv.LinearSolvers
@everywhere using jInv.ForwardShare
@everywhere using jInv.InverseSolve
@everywhere using jInv.Utils
@everywhere using Printf
@everywhere using Helmholtz
@everywhere using jInvSeismic.FWI
@everywhere using jInvSeismic.Utils
@everywhere using jInvSeismic.EikonalInv
@everywhere using jInvSeismic.BasicFWI
@everywhere using Multigrid
using Statistics
using MAT
using DelimitedFiles

@testset "jInvSeismic" begin
	#include("BasicFWI/runtests.jl")
	include("EikonalInv/runtests.jl")
	include("FWI/runtests.jl")
end

