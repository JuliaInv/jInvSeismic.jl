using jInv.InverseSolve
using jInv.LinearSolvers
using Statistics
using jInv.Utils
using Distributed
using DelimitedFiles
using Multigrid.ParallelJuliaSolver


function calculateDobs(m::Array{Float64, 2}, pForp::Array{RemoteChannel},
	omega::Vector, nfreq::Int64)
	Dp, = getData(vec(m),pForp)
	nfreq = length(omega);

	for k=1:length(Dp)
		wait(Dp[k]);
	end
	Dobs = Array{Array}(undef, length(Dp));
	for k=1:length(Dp)
		Dobs[k] = fetch(Dp[k]);
	end

	return Dobs;
end

export solveForwardProblem
function solveForwardProblem(m::Array{Float64, 2}, pForp::Array{RemoteChannel},
	omega::Vector, nfreq::Int64)
	Dobs = calculateDobs(m, pForp, omega, nfreq)
	Wd = Array{Array}(undef, nfreq);

	for k=1:length(Dobs)
		Wd[k] = 1.0./(abs.(real.(Dobs[k])) .+ 1e-1*mean(abs.(Dobs[k])));
	end

	return Dobs, Wd;
end

export solveForwardProblemExtendedSources
function solveForwardProblemExtendedSources(m::Array{Float64, 2}, pForp::Array{RemoteChannel},
	omega::Vector, nfreq::Int64)
	Dobs = calculateDobs(m, pForp, omega, nfreq)
	Wd = Array{Array}(undef, length(Dobs));

	avgDobs = map(dobs_j -> mean(abs.(dobs_j)), Dobs[:]);
	for k=1:length(Dobs)
		Wd[k] = ones(size(Dobs[k])) ./ avgDobs[k] ;
	end

	return Dobs, Wd;
end

export solveForwardProblemNoProcs
function solveForwardProblemNoProcs(m::Array{Float64, 2}, pFor::FWIparam,
	omega::Vector, nfreq::Int64)
	Dp,pFor = getData(vec(m),pFor)
	nfreq = length(omega);
	Dobs = fetch(Dp);
	Wd = 1.0./(Dobs .+ 1e-1*mean(abs.(Dobs)));

	return Dobs, Wd;
end


function wFourthOrderSmoothing(m::Vector, mref::Vector, M::AbstractMesh; Iact=1.0, C=[])
	dm = m.-mref;
	d2R = wdiffusionReg(m,mref,M,Iact = Iact,C = C)[3];
	clear!(M);
	d2R = d2R'*d2R;
	dR  = d2R*dm;
	Rc  = 0.5*dot(dm,dR);
   return Rc,dR,d2R
end

function getFreqContParams(pFor::Array{RemoteChannel}, Dobs::Array, Wd::Array,
	 nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
	boundsHigh::Float64, boundsLow::Float64, plotting::Bool=false,
	plottingFunc::Function=dummy)
	N = prod(Mr.n);
	Iact = sparse(I,N,N);
	mback   = zeros(Float64,N);

	########################################################################################################
	##### Set up remote workers ############################################################################
	########################################################################################################

	Ainv  = getParallelJuliaSolver(Float64,Int64,numCores=4,backend=1);

	# Ainv  = getJuliaSolver();
	## Choose the workers for FWI (here, its a single worker)
	misfun = SSDFun;
	# probsMax = ceil(Integer,nfreq/nworkers());
	nprobs   = zeros(maximum(workers()));
	pMis = getMisfitParam(pFor, Wd, Dobs, misfun, Iact, mback);
	boundsHigh = boundsHigh*ones(Float32,N);
	boundsLow = boundsLow*ones(Float32,N);

	maxStep				=0.05*maximum(boundsHigh);
	# regparams 			= [1.0,1.0,1.0,1e-5];
	cgit 				= 8;
	alpha 				= 1e+1;
	pcgTol 				= 1e-3;
	maxit 				= 1;
	HesPrec 			= getExactSolveRegularizationPreconditioner();
	# regfun(m,mref,M) 	= wdiffusionReg(m,mref,M,Iact=Iact,C=[]);
	regfun(m,mref,M) 	= wFourthOrderSmoothing(m,mref,M,Iact=Iact,C=[]);

	pInv = getInverseParam(Mr,identityMod,regfun,alpha,mref[:],boundsLow,boundsHigh,
	                         maxStep=maxStep,pcgMaxIter=cgit,pcgTol=pcgTol,
							 minUpdate=1e-3, maxIter = maxit,HesPrec=HesPrec);

	function plotIntermediateResults(mc,Dc,iter,pInv,PMis, resultsFilename::String)
		# Models are usually shown in velocity.
		fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
		Temp = splitext(resultsFilename);
		if iter>0
			Temp = string(Temp[1],iter,Temp[2]);
		else
			Temp = resultsFilename;
		end
		if resultsFilename!=""
			DelimitedFiles.writedlm(Temp,convert(Array{Float16},fullMc));
		end
		if plotting
			plottingFunc(fullMc');
		end
	end
	return pInv, pMis, plotIntermediateResults, Iact, mback;
end
#
# export solveInverseProblemExtendedSources
# function solveInverseProblemExtendedSources(pFor::Array{RemoteChannel}, Dobs::Array, Wd::Array,
# 	nfreq::Int64, nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
# 	boundsHigh::Float64, boundsLow::Float64, resultsFilename::String, plotting::Bool=false,
# 	plottingFunc::Function=dummy)
#
# 	regfunFourthOrder(m,mref,M) = wFourthOrderSmoothing(m,mref,M,Iact=Iact,C=[]);
# 	pInv, pMis, plotIntermediateResults, Iact, mback = getFreqContParams(pFor, Dobs, Wd, nfreq, nx, nz, mref, Mr,
# 		boundsHigh, boundsLow, plotting, plottingFunc);
#
# 	regfunDiff(m,mref,M) = wdiffusionReg(m,mref,M,Iact=Iact,C=[]);
#
# 	# Run one sweep of a frequency continuation procedure.
# 	mc, Dc = freqContExtendedSources(copy(mref[:]), pInv, pMis[1:2], 2, 4,Iact,mback, plotIntermediateResults, resultsFilename, 1);
#
# 	pInv.regularizer = regfunDiff;
# 	# Run one sweep of a frequency continuation procedure.
# 	mc, Dc = freqContExtendedSources(copy(mc[:]), pInv, pMis, nfreq, 4,Iact,mback, plotIntermediateResults, resultsFilename, 1);
#
# 	return mc, Dc, pInv, Iact, mback, map(x->fetch(x), pMis);
# end


export solveInverseProblem
function solveInverseProblem(pFor::Array{RemoteChannel}, Dobs::Array, Wd::Array,
	sources::SparseMatrixCSC, sourcesSubInd::Vector,
	contDivFWI::Array{Int64}, nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
	boundsHigh::Float64, boundsLow::Float64, resultsFilename::String, plotting::Bool=false,
	plottingFunc::Function=dummy)

	regfunFourthOrder(m,mref,M) = wFourthOrderSmoothing(m,mref,M,Iact=Iact,C=[]);
	pInv, pMis, plotIntermediateResults, Iact, mback = getFreqContParams(pFor, Dobs, Wd, nx, nz, mref, Mr,
		boundsHigh, boundsLow, plotting, plottingFunc);

	regfunDiff(m,mref,M) = wdiffusionReg(m,mref,M,Iact=Iact,C=[]);
	println("size pmis:" , size(pMis))
	# Run one sweep of a frequency continuation procedure.
	mc, Dc = freqContExtendedSources(copy(mref[:]), sources, sourcesSubInd,
	pInv, pMis, contDivFWI[1:3], 4, resultsFilename, plotIntermediateResults);
	pInv.regularizer = regfunDiff;
	# Run one sweep of a frequency continuation procedure.
	mc, Dc = freqContExtendedSources(copy(mc[:]), sources, sourcesSubInd,
	pInv, pMis, contDivFWI, 4, resultsFilename, plotIntermediateResults);


	# Run one sweep of a frequency continuation procedure.
	# mc, Dc = freqCont(copy(mref[:]),
	# pInv, pMis, contDivFWI[1:3], 4, resultsFilename, plotIntermediateResults);
	# pInv.regularizer = regfunDiff;
	# # Run one sweep of a frequency continuation procedure.
	# mc, Dc = freqCont(copy(mc[:]),
	# pInv, pMis, contDivFWI, 4, resultsFilename, plotIntermediateResults);


	return mc, Dc, pInv, Iact, mback, pMis;
end
#
# export solveInverseProblemTraceEstimation
# function solveInverseProblemTraceEstimation(pFor::Array{RemoteChannel}, Dobs::Array, Wd::Array,
# 	nfreq::Int64, nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
# 	boundsHigh::Float64, boundsLow::Float64, resultsFilename::String, plotting::Bool=false,
# 	plottingFunc::Function=dummy)
# 	pInv, pMis, plotIntermediateResults, Iact, mback = getFreqContParams(pFor,
# 		Dobs, Wd, nfreq, nx, nz, mref, Mr,
# 		boundsHigh, boundsLow, plotting, plottingFunc);
#
# 	# Run one sweep of a frequency continuation procedure.
# 	mc, Dc = freqContTraceEstimation(copy(mref[:]), pInv, pMis, nfreq, 4,
# 			Iact, mback, plotIntermediateResults, resultsFilename, 1);
#
# 	return mc, Dc, pInv, Iact, mback;
# end

export solveInverseProblemNoProcs
function solveInverseProblemNoProcs(pFor::FWIparam, Dobs::Array, Wd::Array, nfreq::Int64,
	nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
	boundsHigh::Float64, boundsLow::Float64, plotting::Bool=false, plottingFunc::Function=dummy)
	N = prod(Mr.n);
	Iact = sparse(I,N,N);
	mback   = zeros(Float64,N);

	########################################################################################################
	##### Set up remote workers ############################################################################
	########################################################################################################

	Ainv = getJuliaSolver();
	## Choose the workers for FWI (here, its a single worker)
	misfun = SSDFun;
	# probsMax = ceil(Integer,nfreq/nworkers());
	nprobs   = zeros(maximum(workers()));

	pMis = getMisfitParam(pFor, Wd, Dobs, misfun);

	boundsHigh = boundsHigh*ones(Float32,N);
	boundsLow = boundsLow*ones(Float32,N);

	maxStep				=0.05*maximum(boundsHigh);

	#regparams 			= [1.0,1.0,1.0,1e-5];
	cgit 				= 8;
	alpha 				= 1e+2;
	pcgTol 				= 1e-1;
	maxit 				= 30;
	HesPrec 			= getExactSolveRegularizationPreconditioner();
	regfun(m,mref,M) 	= wdiffusionReg(m,mref,M,Iact=Iact,C=[]);


	pInv = getInverseParam(Mr,identityMod,regfun,alpha,mref[:],boundsLow,boundsHigh,
	                         maxStep=maxStep,pcgMaxIter=cgit,pcgTol=pcgTol,
							 minUpdate=1e-3, maxIter = maxit,HesPrec=HesPrec);


	 function plotIntermediateResults(mc,Dc,iter,pInv,PMis)
	 	# Models are usually shown in velocity.
	 	fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
	 	if plotting
			plottingFunc(fullMc');
	 	end
	 end

	mc, Dc, = projGNCG(copy(mref[:]),pInv,pMis);
	return mc, Dc, pInv, Iact, mback;
end
