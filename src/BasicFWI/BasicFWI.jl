module BasicFWI
using jInv.Mesh
using jInv.Utils
using Distributed
using SparseArrays
using LinearAlgebra
using DelimitedFiles

export BasicFWIparam
import jInv.ForwardShare.ForwardProbType
"""
	BasicFWIparam <: ForwardProbType

	type defining Full Waveform Inversion (FWI) problem.

	Constructor: getBasicFWIparam(omega,gamma,Q,P,Mesh)

	Fields:
		omega     - frequencies
		gamma     - attenuation
		Sources   - matrix of sources
		Receivers - receiver matrix
		Mesh      - forward mesh
		Fields    -
		Ainv      - for linear solver, LU factorizations
"""
mutable struct BasicFWIparam <: ForwardProbType
    omega     # frequencies
    gamma     # attenuation
    Sources   # Sources
    Receivers # Receivers
    Mesh      # Mesh
	Fields    # Fields
	# Helmholtz # Helmholtz operators
	Ainv      # LU factorization
	originalSources	  # sources before trace estimation
	ExtendedSources
end

export getBasicFWIparam
"""
	pFor = getBasicFWIparam(omega,gamma,Q,P,Mesh,doDistribute=false)

	constructs BasicFWI param

"""
function getBasicFWIparam(omega,gamma,Q,P,Mesh,doDistribute=false)
    nfreq = length(omega)

    if doDistribute && (nfreq > 1)
        pFor     = Array{RemoteChannel}(undef, nfreq)
        probsMax = ceil(Integer,nfreq/nworkers())
        nprobs   = zeros(maximum(workers()))

        i=1; nextidx() = (idx = i; i+=1; idx)

        @sync begin
            for p=workers()
                @async begin
                    while true
                        if nprobs[p]>=probsMax
                            break
                        end
                        idx = nextidx()
                        if (idx>nfreq)
                            break
                        end
                        pFor[idx] = initRemoteChannel(getBasicFWIparam,p,omega[idx],gamma,Q,P,Mesh)
                        nprobs[p] +=1
                        wait(pFor[idx])
                    end
                end
            end
        end
    else
        pFor = BasicFWIparam(omega,gamma,Q,P,Mesh,[],[],Q,Q)
    end
    return pFor
end

include("getData.jl")
include("getSensMatVec.jl")
include("getSensTMatVec.jl")
include("getHelmholtzOperator.jl")
include("getMassMatrix.jl")
include("freqCont.jl")
include("solvers.jl")

end
