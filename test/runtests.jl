using Test
using Distributed

if nworkers() == 1
	addprocs(2);
elseif nworkers() < 2
	addprocs(2 - nworkers());
end

include("BasicFWI/runtests.jl")
include("EikonalInv/runtests.jl")
include("FWI/runtests.jl")

