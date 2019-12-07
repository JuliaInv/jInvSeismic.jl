export freqContBasic;
export freqContTraceEstimation;
using jInv.InverseSolve
using Printf
using KrylovMethods
global inx = 1;
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

	Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, map(pm -> fetch(pm), pMisTemp));
	println("Misfit for freq: ", freqIdx, " after gn5 is: ", F);
	# println("Misfit at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end
return mc,Dc,flag,HIS;
end

struct NsParams
    source
    GNIter
    FCIter
end

function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, params::NsParams, HinvPs::Vector{Array}, beta::Float64)
	recvSize = size(pMisArr[1].pFor.Receivers)
	s = params.source;
	println("starting: ", s);

	numOfCurrentProblems = size(currentProblems, 1);
	Ap = Vector{Matrix}(undef, numOfCurrentProblems);
	diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
	B = Vector{Vector}(undef, numOfCurrentProblems);
	# ref = zeros(sizeH[1])
	ref = Vector(pMisArr[1].pFor.originalSources[:, s]);
	for i=1:numOfCurrentProblems
		WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
		Ap[i] = HinvPs[i]
		diags[i] =  WdSqr;
		B[i] =  2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvPs[i] * WdSqr * pMisArr[i].dobs[:,s,1];
	end

	newSource = KrylovMethods.cg((x-> real(sum(map((Ai, diag) ->  Ai * diag * (Ai'* x) + 2*beta.* x, Ap, diags)))), real(sum(B)), x=ref, tol=1e-8, maxIter=100)[1];
	# writeSource(newSource, s);
	# function writeSource(source, sourceIdx)
	if s < 10
		writedlm(string("Ns", s, "_GN", params.GNIter, "_FC", params.FCIter), newSource);
	end
	# end


	return newSource;
end
function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, nsParams::Vector{NsParams}, HinvPs::Vector{Array},
	beta::Float64)
	newSources = Array{Array}(undef, size(nsParams))
	for (i,params) in enumerate(nsParams)
		newSources[i] = calculateZs(currentProblems, sizeH, pMisArr, params, HinvPs,
		beta);
	end
	return newSources;
end

function minimizeZs(mc, currentProblems::UnitRange, HinvPs::Vector{Array},
	sizeH::Tuple, pMisArr::Array{MisfitParam}, nsrc::Integer, beta::Float64,
	runningProcs::Array, GNIter::Int, FCIter::Int)
	pMisTemp = Array{RemoteChannel}(undef, length(currentProblems));
	t111 = time_ns();
	newSourcesp = Array{RemoteChannel}(undef, nworkers());
	@sync begin
		for worker in 1:nworkers()
			@async begin
				newSourcesp[worker] = initRemoteChannel(calculateZs, workers()[worker],
				currentProblems, sizeH, pMisArr,
				map(s-> NsParams(s, GNIter, FCIter),worker:nworkers():nsrc),
				HinvPs, beta);
			end
		end
	end
	newSources = Array{Array}(undef, nworkers());
	# Qs = pMisArr[1].pFor.originalSources;
	# s1 = size(Qs)[1];
	for worker in 1:nworkers()
		newSources[worker] = fetch(newSourcesp[worker])
		println("size for worker: ", size(newSources[worker]));
	end
	for i=1:size(currentProblems, 1)
		# Sources = Matrix{ComplexF64}(undef, s1,nsrc);
		for s=1:nsrc
			# pMisArr[i].pFor.ExtendedSources[:, s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];
			pMisArr[i].pFor.Sources[:, s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];
			# Sources[:,s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];

		end
		# pMisArr[i].pFor.Sources = Sources;
		pMisTemp[i] = initRemoteChannel(x->x, runningProcs[i], pMisArr[i]);
	end

	s111 = time_ns();
	println("FREQCONT ZS");
	println((s111 - t111)/1.0e9);
	return pMisTemp;
end

function freqContZs(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
	Iact,mback::Union{Vector,AbstractFloat,AbstractModel, Array{Float64,1}},
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];
pFor = fetch(pMis[1]).pFor
Z = copy(pFor.originalSources);
nrec = size(pFor.Receivers, 2);
sizeH = size(pFor.Ainv[1]);
pFor = nothing
nsrc = size(Z, 2);
beta = 1e-4;
for freqIdx = startFrom:nfreq
	println("start freqCont Zs iteration from: ", freqIdx)
	tstart = time_ns();
	reqIdx1 = freqIdx;
	if freqIdx > 1
		reqIdx1 = max(1,freqIdx-windowSize+1);
	end
	reqIdx2 = freqIdx;
	currentProblems = reqIdx1:reqIdx2;


	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");


	for j=1:5
		pMisCurrent = map(fetch, pMis[currentProblems]);
		pForpCurrent =  map(x->x.pFor, pMisCurrent);
		Dp,pForp = getData(vec(mc), pForpCurrent);

		pForCurrent = map(x->fetch(x), pForp);
		numOfCurrentProblems = size(currentProblems, 1);
		map((pm,pf) -> pm.pFor = pf , pMisCurrent, pForCurrent);
		HinvPs = Vector{Array}(undef, numOfCurrentProblems);

		t1 = time_ns();
		for freqs = 1:numOfCurrentProblems
			HinvPs[freqs] = (pForCurrent[freqs].Ainv[1])' \ Matrix(pForCurrent[freqs].Receivers);
			println("HINVP done");
		end
		e1 = time_ns();
		println("runtime of HINVPs");
		println((e1 - t1)/1.0e9);

		if resultsFilename == ""
				filename = "";
			else
				Temp = splitext(resultsFilename);
				filename = string(Temp[1],"_FC",freqIdx,"_",j,"_GN",Temp[2]);
		end

		function dumpGN(mc,Dc,iter,pInv,PF)
			dumpFun(mc,Dc,iter,pInv,PF,filename);
		end

		# function writeSource(source, sourceIdx)
		# 	if sourceIdx < 10
		# 		writedlm(string("Ns", sourceIdx, "_GN", j, "_FC", freqIdx), source)
		# 	end
		# end

		t2 = time_ns();
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);
		e2 = time_ns();
		println("Misfit B4 mzs at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
		println("runtime of compute misfit");
		println((e2 - t2)/1.0e9);

		t3 = time_ns();
		pMisTemp = minimizeZs(mc, currentProblems, HinvPs,sizeH, pMisCurrent, nsrc, beta,
		map(x->x.where, pMis[currentProblems]), j, freqIdx);

		e3 = time_ns();
		println("runtime of minimizeZs");
		println((e3 - t3)/1.0e9);



		#
		# runningProcs = map(x->x.where, pMisTemp);
		# pMisCurrent = map(fetch, pMisTemp);
		# pForpCurrent =  map(x->x.pFor, pMisCurrent);
		# eta = 20.0;
		# Dobs = map(x->x.dobs[:,:], pMisCurrent);
		# Wd = map(x->x.Wd[:,:], pMisCurrent);
		# DobsNew = copy(Dobs[:]);
		# nsrc = size(DobsNew[1],2);
		# TEmat = rand([-1,1],(nsrc,10));
		# WdEta = eta.*Wd;
		# DpNew, = getData(vec(mc),pForpCurrent);
		# DobsTemp = Array{Array}(undef, numOfCurrentProblems);
		# WdNew = Array{Array}(undef, numOfCurrentProblems);
		# for i=1:numOfCurrentProblems
		# 	DobsTemp[i] = fetch(DpNew[i]);
		# 	for s=1:nsrc
		# 		etaM1 = 2 .*diagm(0 => vec(Wd[1][:,1])).*diagm(0 => vec(Wd[1][:,1]));
		# 		println("SIZE OLD ETAM: ");
		# 		println(size(etaM1));
		# 		etaM = 2 * eta .*diagm(0 => vec(Wd[i][:,s])).*diagm(0 => vec(Wd[i][:,s]));
		# 		println("SIZE NEW ETAM: ");
		# 		println(size(etaM));
		# 		A = (2 .*I + etaM)
		# 		b = (etaM)*(A\Dobs[i][:,s])
		# 		DobsNew[i][:,s] = 2 .* (A\DobsTemp[i][:,s]) + b;
		# 	end
		# 	DobsNew[i] = DobsNew[i][:,:] * TEmat;
		# 	WdNew[i] = ones((size(DobsNew[i], 1), 10))./sqrt(10);
		# 	println("SIZE OLD WD: ");
		# 	println(size(WdNew[i]));
		# 	WdNew[i] = 1.0./(abs.(real.(DobsNew[i])) .+ 1e-1*mean(abs.(DobsNew[i])));
		# 	WdNew[i] = WdNew[i] ./ sqrt(10);
		# 	println("SIZE NEW WD: ");
		# 	println(size(WdNew[i]));
		# end
		#
		# pForCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
		# for i=1:numOfCurrentProblems
		# 	pForTemp = pForpCurrent[i];
		# 	pForTemp.Sources = pForTemp.ExtendedSources * TEmat;
		# 	pForCurrent[i] = initRemoteChannel(x->x, runningProcs[i], pForTemp);
		# end
		#
		#
		# pMisTemp2 = getMisfitParam(pForCurrent, WdNew, DobsNew, SSDFun, Iact, mback);
		#
		# println("size of exsources:" , size(fetch(pMisTemp2[1]).pFor.Sources));
		#
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, map(pm -> fetch(pm), pMisTemp));

		println("Misfit at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);



		pInv.mref = mc[:];
		t4 = time_ns();
		mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);
		e4 = time_ns();
		println((t4 - e4)/1.0e9);
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, map(pm -> fetch(pm), pMisTemp));

		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);

		pMis[currentProblems] = pMisTemp;
		clear!(pMisTemp);
	end
	beta *= 10;

	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
	global inx = inx + 1;

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
			etaM1 = 2 .*diagm(0 => vec(Wd[1][:,1])).*diagm(0 => vec(Wd[1][:,1]));
			println("SIZE OLD ETAM: ");
			println(size(etaM1));
			etaM = 2 * eta .*diagm(0 => vec(Wd[i][:,s])).*diagm(0 => vec(Wd[i][:,s]));
			println("SIZE NEW ETAM: ");
			println(size(etaM));
			A = (2 .*I + etaM)
			b = (etaM)*(A\Dobs[i][:,s])
			DobsNew[i][:,s] = 2 .* (A\DobsTemp[i][:,s]) + b;
		end
		DobsNew[i] = DobsNew[i][:,:] * TEmat;
		WdNew[i] = ones((size(DobsNew[i], 1), 10))./sqrt(10);
		println("SIZE OLD WD: ");
		println(size(WdNew[i]));
		WdNew[i] = 1.0./(abs.(real.(DobsNew[i])) .+ 1e-1*mean(abs.(DobsNew[i])));
		WdNew[i] = WdNew[i] ./ sqrt(10);
		println("SIZE NEW WD: ");
		println(size(WdNew[i]));
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

function dummy(mc,Dc,iter,pInv,pMis)
# this function does nothing and is used as a default for dumpResults().
end
