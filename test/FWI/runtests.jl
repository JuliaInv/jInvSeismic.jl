@testset "FWI" begin
	include("testGetData.jl")
	include("testSensitivity.jl")
	include("testInversion.jl")
	#include("testTimeDomain.jl")
end

