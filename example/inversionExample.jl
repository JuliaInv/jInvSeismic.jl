include("../test/BasicFWI/setupFWItests.jl");

@everywhere begin
using jInv.InverseSolve
using jInv.LinearSolvers
end

using jInvVis
using PyPlot

close("all")
tstart = time_ns();

function plotInputData()
	figure(20)
	imshow(m');colorbar();

	figure(21);
	imshow(reshape(gamma,(nx,nz))');
end

function solveForwardProblem(pForp::Array{RemoteChannel}, omega::Vector, nrec::Int64,
	nsrc::Int64, nfreq::Int64)
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

function solveInverseProblem(Dobs::Array, Wd::Array, nfreq::Int64, nx::Int64, nz::Int64,
	Mr::RegularMesh)
	mref = 3.0*ones(nx,nz);
	mref = 1.0./(mref.^2); # convert to slowness squared
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
	pForFreq = Array{FWIparam}(undef, nfreq);

	probsMax = ceil(Integer,nfreq/nworkers());
	nprobs   = zeros(maximum(workers()));

	pMis = getMisfitParam(pForp, Wd, Dobs, misfun, Iact, mback);

	boundsHigh = 0.12*ones(Float32,N);
	boundsLow = 0.035*ones(Float32,N);

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


	plotting = true;
	function plotIntermediateResults(mc,Dc,iter,pInv,PMis)
		# Models are usually shown in velocity.
		fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
		if plotting
			close(888);
			figure(888);
			imshow(fullMc');colorbar();
			pause(1.0)
		end
	end

	# Run one sweep of a frequency continuation procedure.
	mc, = freqCont(copy(mref[:]), pInv, pMis, nfreq, 4,plotIntermediateResults,1);
	return mc, pInv, Iact, mback;
end

plotInputData();
Dobs, Wd = solveForwardProblem(pForp, omega, nrec, nsrc, nfreq);
mc, pInv, Iact, mback = solveInverseProblem(Dobs, Wd, nfreq, nx, nz, Mr);



fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
println("Model error:");
println(norm(fullMc.-m));
tend = time_ns();
println("Runtime:");
println((tend - tstart)/1.0e9);

#Plot residuals
figure(22);
imshow((abs.(m.-fullMc))'); colorbar();
