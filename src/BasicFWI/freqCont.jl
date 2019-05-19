export freqContBasic;
export freqContTraceEstimation;
using jInv.InverseSolve

"""
	function freqContBasic
	Frequency continuation procedure for running FWI.
	This function runs GaussNewton on misfit functions defined by pMis with nfreq frequencies.

	Input:
		mc    		- current model
		pInv		- Inverse param
		pMis 		- misfit params (remote)
		nfreq		- number of frequencies in the problem
		windowSize  - How many frequencies to treat at once at the most.
		dumpFun     - a function for plotting, saving and doing all the things with the intermidiate results.
		resultsFilename - file name to save results
		startFrom   - a start index for the continuation. Usefull when our run broke in the middle of the iterations.

"""
function freqContBasic(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];
for freqIdx = startFrom:nfreq
	println("start freqCont iteration from: ", freqIdx)
	tstart = time_ns();

	reqIdx1 = freqIdx;
	if freqIdx > 1
		reqIdx1 = max(1,freqIdx-windowSize+1);
	end
	reqIdx2 = freqIdx;
	currentProblems = reqIdx1:reqIdx2;
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	pMisTemp = pMis[currentProblems];
	pInv.mref = mc[:];

	if resultsFilename == ""
			filename = "";
		else
			Temp = splitext(resultsFilename);
			filename = string(Temp[1],"_FC",freqIdx,"_GN",Temp[2]);
	end
	# Here we set a dump function for GN for this iteracion of FC
	function dumpGN(mc,Dc,iter,pInv,PF)
		dumpFun(mc,Dc,iter,pInv,PF,filename);
	end

	mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);

	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end
return mc,Dc,flag,HIS;
end

"""
	function freqContTraceEstimation
	Frequency continuation procedure for running FWI.
	this function first reduces the number of sources by trace estimation,
	then minimizing the misfit using alternating minimization between dsj and m

	Input:
		mc    		- current model
		pInv		- Inverse param
		pMis 		- misfit params (remote)
		nfreq		- number of frequencies in the problem
		windowSize  - How many frequencies to treat at once at the most.
		Iact        - Projector to active cells.
		sigmaBack   - Background model ("frozen" cells).
		dumpFun     - a function for plotting, saving and doing all the things with the intermidiate results.
		resultsFilename - file name to save results
		startFrom   - a start index for the continuation. Usefull when our run broke in the middle of the iterations.

"""
function freqContTraceEstimation(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
			Iact,mback::Union{Vector,AbstractFloat,AbstractModel},
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];

for freqIdx = startFrom:nfreq
	println("start freqCont iteration from: ", freqIdx)
	tstart = time_ns();

	reqIdx1 = freqIdx;
	if freqIdx > 1
		reqIdx1 = max(1,freqIdx-windowSize+1);
	end
	reqIdx2 = freqIdx;
	currentProblems = reqIdx1:reqIdx2;
	numOfCurrentProblems = size(currentProblems, 1);
	runningProcs = map(x->x.where, pMis[currentProblems]);
	pMisCurrent = map(fetch, pMis[currentProblems]);
	pForpCurrent =  map(x->x.pFor, pMisCurrent);
	eta = 20.0;
	Dobs = map(x->x.dobs[:,:], pMisCurrent);
	Wd = map(x->x.Wd[:,:], pMisCurrent);
	DobsNew = copy(Dobs[:]);
	nsrc = size(DobsNew[1],2);
	TEmat = rand([-1,1],(nsrc,10));
	WdEta = eta.*Wd;
	DpNew, = getData(vec(mc),pForpCurrent);
	DobsTemp = Array{Array}(undef, numOfCurrentProblems);
	WdNew = Array{Array}(undef, numOfCurrentProblems);
	for i=1:numOfCurrentProblems
		DobsTemp[i] = fetch(DpNew[i]);
		for s=1:nsrc
			etaM = 2 .*diagm(0 => vec(Wd[1][:,1])).*diagm(0 => vec(Wd[1][:,1]));
			A = (2 .*I + etaM)
			b = (etaM)*(A\Dobs[i][:,s])
			DobsNew[i][:,s] = 2 .* (A\DobsTemp[i][:,s]) + b;
		end
		DobsNew[i] = DobsNew[i][:,:] * TEmat;
		WdNew[i] = ones((size(DobsNew[i], 1), 10))./sqrt(10);
	end

	pForCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
	for i=1:numOfCurrentProblems
		pForTemp = pForpCurrent[i];
		pForTemp.Sources = pForTemp.originalSources * TEmat;
		pForCurrent[i] = initRemoteChannel(x->x, runningProcs[i], pForTemp);
	end
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	pMisTemp = getMisfitParam(pForCurrent, WdNew, DobsNew, SSDFun, Iact, mback);
	pInv.mref = mc[:];

	if resultsFilename == ""
			filename = "";
		else
			Temp = splitext(resultsFilename);
			filename = string(Temp[1],"_FC",freqIdx,"_GN",Temp[2]);
	end
	# Here we set a dump function for GN for this iteracion of FC
	function dumpGN(mc,Dc,iter,pInv,PF)
		dumpFun(mc,Dc,iter,pInv,PF,filename);
	end

	mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);

	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end

return mc,Dc,flag,HIS;
end
