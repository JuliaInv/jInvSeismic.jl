using jInv.InverseSolve
using jInv.LinearSolvers

export solveForwardProblem
function solveForwardProblem(m::Array{Float64, 2}, pForp::Array{RemoteChannel}, omega::Vector, nrec::Int64,
	nsrc::Int64, nfreq::Int64)

	println("WTF2")
	Dp,pFor = getData(vec(m),pForp)
	nfreq = length(omega);

	for k=1:length(Dp)
		wait(Dp[k]);
	end
	Dobs = Array{Array}(undef, nfreq);
	for k=1:length(Dp)
		Dobs[k] = fetch(Dp[k]);
	end
	Wd = Array{Array}(undef, nfreq);
	for k=1:length(Dobs)
		Wd[k] = 1.0./(Dobs[k] .+ 1e-2*maximum(abs.(Dobs[k])));
	end

	return Dobs, Wd;
end

export solveForwardProblemNoProcs
function solveForwardProblemNoProcs(m::Array{Float64, 2}, pFor::FWIparam, omega::Vector, nrec::Int64,
	nsrc::Int64, nfreq::Int64)
	Dp,pFor = getData(vec(m),pFor)
	nfreq = length(omega);
	Dobs = fetch(Dp);
	Wd = 1.0./(Dobs .+ 1e-2*maximum(abs.(Dobs)));

	return Dobs, Wd;
end

export solveInverseProblem
function solveInverseProblem(pFor::Array{RemoteChannel}, Dobs::Array, Wd::Array,
	nfreq::Int64, nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
	boundsHigh::Float64, boundsLow::Float64, resultsFilename::String, plotting::Bool=false,
	plottingFunc::Function=dummy)
	println("SAGIB UPDATE 4");
	# mref = 3.0*ones(nx,nz);
	# mref = 1.0./(mref.^2); # convert to slowness squared
	N = prod(Mr.n);
	Iact = sparse(I,N,N);
	mback   = zeros(Float64,N);

	########################################################################################################
	##### Set up remote workers ############################################################################
	########################################################################################################

	Ainv = getJuliaSolver();
	## Choose the workers for FWI (here, its a single worker)
	misfun = SSDFun;
	pMis = Array{MisfitParam}(undef, nfreq);
	pForFreq = Array{BasicFWIparam}(undef, nfreq);

	probsMax = ceil(Integer,nfreq/nworkers());
	nprobs   = zeros(maximum(workers()));





	println("size dobs: ", size(Dobs));
	pMis = getMisfitParam(pFor, Wd, Dobs, misfun, Iact, mback);
	println("pforK size: ", size(pFor));
	println("pmis size: ", size(pMis));
	boundsHigh = boundsHigh*ones(Float32,N);
	boundsLow = boundsLow*ones(Float32,N);

	maxStep				=0.05*maximum(boundsHigh);

	# regparams 			= [1.0,1.0,1.0,1e-5];
	cgit 				= 8;
	alpha 				= 1e+1;
	pcgTol 				= 1e-1;
	maxit 				= 5;
	HesPrec 			= getExactSolveRegularizationPreconditioner();
	regfun(m,mref,M) 	= wdiffusionReg(m,mref,M,Iact=Iact,C=[]);


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

	println("print FWI");
	# Run one sweep of a frequency continuation procedure.
	mc, Dc = freqCont(copy(mref[:]), pInv, pMis, nfreq, 4, plotIntermediateResults, resultsFilename, 1);

	# mc, Dc = freqCont(copy(mref[:]),pFor, Dobs, Wd, pInv, misfun, Iact, mback, nfreq, 4, plotIntermediateResults, resultsFilename, 1);
	return mc, Dc, pInv, Iact, mback;
end

export solveInverseProblemNoProcs
function solveInverseProblemNoProcs(pFor::FWIparam, Dobs::Array, Wd::Array, nfreq::Int64,
	nx::Int64, nz::Int64, mref::Array{Float64,2}, Mr::RegularMesh,
	boundsHigh::Float64, boundsLow::Float64, plotting::Bool=false, plottingFunc::Function=dummy)
	# mref = 3.0*ones(nx,nz);
	# mref = 1.0./(mref.^2); # convert to slowness squared
	N = prod(Mr.n);
	Iact = sparse(I,N,N);
	mback   = zeros(Float64,N);

	########################################################################################################
	##### Set up remote workers ############################################################################
	########################################################################################################

	Ainv = getJuliaSolver();
	## Choose the workers for FWI (here, its a single worker)
	misfun = SSDFun;
	# pMis = Array{MisfitParam}(undef, nfreq);
	# pForFreq = Array{FWIparam}(undef, nfreq);

	probsMax = ceil(Integer,nfreq/nworkers());
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

	# Run one sweep of a frequency continuation procedure.
	mc, Dc = freqContBasic(copy(mref[:]), pInv, pMis, nfreq, 4, plotIntermediateResults,1);
	#mc, Dc, = projGNCG(copy(mref[:]),pInv,pMis);
	return mc, Dc, pInv, Iact, mback;
end
