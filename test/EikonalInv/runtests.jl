
using Distributed
if nworkers()<2
	addprocs(2-nworkers())
end

using Test
using LinearAlgebra
using SparseArrays
@everywhere using jInv.Mesh
@everywhere using jInv.ForwardShare
@everywhere using jInv.Utils
@everywhere using jInvSeismic.EikonalInv
@everywhere using jInv.InverseSolve


@testset "EikonalInv" begin
	include("testGetData2D.jl")
	include("testGetData3D.jl")
	include("testSensMatVec.jl")
	include("testTravelTimeInversion.jl")
end