module FWI

using SparseArrays
using LinearAlgebra
using Distributed

using jInv.Mesh
using jInv.Utils
using jInv.LinearSolvers
using jInv.InverseSolve
using jInvSeismic.Utils
using KrylovMethods
using Multigrid
using Helmholtz
using MAT
using FFTW

import jInv.ForwardShare.getData
import jInv.ForwardShare.getSensTMatVec
import jInv.ForwardShare.getSensMatVec
import jInv.LinearSolvers.copySolver

import jInv.ForwardShare.ForwardProbType

FieldsType = ComplexF64

useSommerfeldBC = true;

fieldsFilenamePrefix = "tempFWIfields"

function getFieldsFileName(omega::Float64)
	omRound = string(round((omega/(2*pi))*100.0)/100.0);
	tfilename = string(fieldsFilenamePrefix,"_f",omRound,"_worker",myid(),".mat");
end


export clear!
export setSourceSelection,setSourceSelectionRatio,setSourceSelectionNum

export FWIparam, getFWIparam, getFWIparamFreqOnlySplit
mutable struct FWIparam <: ForwardProbType
    omega					:: Float64     # frequencies
	WaveletCoef				:: ComplexF64
    gamma					:: Vector{Float64}     # attenuation
    Sources					:: Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}, Array{ComplexF64, 2}}   # Sources
	OriginalSources			:: Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}, Array{ComplexF64, 2}}   # Sources
    Receivers				:: Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}}
	Fields					:: Array{FieldsType}
    Mesh      				:: RegularMesh
	ForwardSolver			:: AbstractSolver
	forwardSolveBatchSize	:: Int64
	sourceSelection			:: Array{Int64,1}
	useFilesForFields		:: Bool
end

function getFWIparam(omega::Float64, WaveletCoef::ComplexF64, gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Mesh::RegularMesh, ForwardSolver:: AbstractSolver, workerList::Array{Int64},forwardSolveBatchSize::Int64=size(Sources,2),useFilesForFields::Bool = false)
	return getFWIparam([omega], [WaveletCoef],gamma,Sources,Receivers, Mesh,ForwardSolver, workerList,forwardSolveBatchSize,useFilesForFields);
end

function getFWIparam(omega::Array{Float64}, WaveletCoef::Array{ComplexF64},gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Mesh::RegularMesh, ForwardSolver::AbstractSolver, workerList::Array{Int64},forwardSolveBatchSize::Int64=size(Sources,2),useFilesForFields::Bool = false)

	continuationDivision = zeros(Int64,length(omega)+1);
	continuationDivision[1] = 1;

	if workerList==[]
		ActualWorkers = workers();
	else
		ActualWorkers = intersect(workerList,workers());
		if length(ActualWorkers)<length(workerList)
			warn("FWI: workerList included indices of non-existing workers.")
		end
	end
	numWorkers = length(ActualWorkers);
	pFor   = Array{RemoteChannel}(undef,numWorkers*length(omega));
	SourcesSubInd = Array{Array{Int64,1}}(undef,numWorkers*length(omega));
	for k=1:length(omega)
		getFWIparamInternal(omega[k],WaveletCoef[k], gamma,Sources,Receivers,zeros(FieldsType,0), Mesh,
									ForwardSolver, forwardSolveBatchSize ,ActualWorkers,pFor,(k-1)*numWorkers+1,SourcesSubInd,useFilesForFields);
		continuationDivision[k+1] = k*numWorkers+1;
	end
	return pFor,continuationDivision,SourcesSubInd # Array of Remote Refs
end

function getFWIparamFreqOnlySplit(omega::Float64, WaveletCoef::ComplexF64, gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Mesh::RegularMesh, ForwardSolver:: AbstractSolver, workerList::Array{Int64},forwardSolveBatchSize::Int64=size(Sources,2),useFilesForFields::Bool = false)
	return getFWIparamFreqOnlySplit([omega], [WaveletCoef],gamma,Sources,Receivers, Mesh,ForwardSolver, workerList,forwardSolveBatchSize,useFilesForFields);
end

function getFWIparamFreqOnlySplit(omega::Array{Float64}, WaveletCoef::Array{ComplexF64},gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Mesh::RegularMesh, ForwardSolver::AbstractSolver, workerList::Array{Int64},forwardSolveBatchSize::Int64=size(Sources,2),useFilesForFields::Bool = false)
	if workerList==[]
		ActualWorkers = workers();
	else
		ActualWorkers = intersect(workerList,workers());
		if length(ActualWorkers)<length(workerList)
			warn("FWI: workerList included indices of non-existing workers.")
		end
	end
	numWorkers = length(ActualWorkers);
	pFor   = Array{RemoteChannel}(undef,length(omega));
	for k=1:length(omega)
		getFWIparamInternalFreqOnly(omega[k],WaveletCoef[k], gamma,Sources,Receivers,zeros(FieldsType,0), Mesh,
									ForwardSolver, forwardSolveBatchSize ,ActualWorkers[((k-1) % numWorkers) + 1],pFor,k,useFilesForFields);
	end
	return pFor,0,0 # Array of Remote Refs
end

function getFWIparamInternalFreqOnly(omega::Float64, WaveletCoef::ComplexF64,gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							fields::Array{FieldsType}, Mesh::RegularMesh,
							ForwardSolver:: AbstractSolver, forwardSolveBatchSize::Int64,
							Workers::Int64, pFor::Array{RemoteChannel},startPF::Int64,useFilesForFields::Bool = false)
	nsrc  = size(Sources,2);
	pFor[startPF] = initRemoteChannel(getFWIparamInternal,Workers, omega,WaveletCoef,  gamma, Sources, Receivers, fields, Mesh,
															copySolver(ForwardSolver),forwardSolveBatchSize,useFilesForFields);
	wait(pFor[startPF]);
	return pFor # Array of Remote Refs
end

function getFWIparamInternal(omega::Float64, WaveletCoef::ComplexF64,gamma::Vector{Float64},
							Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							fields::Array{FieldsType}, Mesh::RegularMesh,
							ForwardSolver:: AbstractSolver, forwardSolveBatchSize::Int64,
							Workers::Array{Int64}, pFor::Array{RemoteChannel},startPF::Int64,SourcesSubInd::Array{Array{Int64,1},1},useFilesForFields::Bool = false)
	i = startPF; nextidx() = (idx=i; i+=1; idx)
	nsrc  = size(Sources,2);
	numWorkers = length(Workers);
	# send out jobs
	@sync begin
		for p=Workers
			@async begin
				while true
					idx = nextidx()
					if idx > startPF + numWorkers - 1
						break
					end
					I_k = getSourcesIndicesOfKthWorker(numWorkers,idx - startPF + 1,nsrc)
					SourcesSubInd[idx] = I_k;
					pFor[idx] = initRemoteChannel(getFWIparamInternal,p, omega,WaveletCoef,  gamma, Sources[:,I_k], Receivers, fields, Mesh,
																			copySolver(ForwardSolver),forwardSolveBatchSize,useFilesForFields);
					wait(pFor[idx]);
				end
			end
		end
	end
	return pFor # Array of Remote Refs
end

function getFWIparamInternal(omega::Float64,WaveletCoef::ComplexF64,
							gamma::Vector{Float64},
	                        Sources::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Receivers::Union{Vector{Float64},SparseMatrixCSC,Array{Float64,2}},
							Fields::Array{FieldsType},
							Mesh::RegularMesh, ForwardSolver:: AbstractSolver, forwardSolveBatchSize::Int64,useFilesForFields::Bool = false)
	return FWIparam(omega,WaveletCoef,gamma,Sources,Sources,Receivers,Fields,Mesh,ForwardSolver,forwardSolveBatchSize,Array{Int64}(undef,0),useFilesForFields)
end

import jInv.Utils.clear!
function clear!(pFor::FWIparam)
	clear!(pFor.ForwardSolver);
	pFor.Fields = zeros(FieldsType,0);
	clear!(pFor.Mesh);
	return pFor;
end

include("getData.jl")
include("getSensMatVec.jl")
include("getSensTMatVec.jl")
include("FourthOrderHesPrec.jl")
include("freqCont.jl")
include("timeDomainFWI.jl")
include("firstArivalPicking.jl")
include("computeHinvTRec.jl")
end
