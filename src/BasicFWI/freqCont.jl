export freqContBasic;
export freqContTraceEstimation;
using jInv.InverseSolve
# using JLD
using KrylovMethods

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
	println("Misfit for freq: ", freqIdx, " after 5 GN iterations is: ", F);
	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end
return mc,Dc,flag,HIS;
end

function calculateZ2(misfitCalc::Function, p::Integer, nsrc::Integer,
	numOfCurrentProblems::Integer, Wd::Array, HinvPs::Array,
	pMisCurrent::Array{MisfitParam}, Z1::Matrix, alpha::Float64)

	println("misfit at start:: ", misfitCalc())
	rhs = zeros(ComplexF64, (p, nsrc));
	for i = 1:numOfCurrentProblems
		pm = pMisCurrent[i]
		println("size wd: ", mean(Wd[i]))

		rhs += (mean(Wd[i])^2) .* Z1' * HinvPs[i] * (-HinvPs[i]' * pm.pFor.Sources + pm.dobs[:,:,1]);
	end
	lhs = zeros(ComplexF64, (p,p));
	println("size Wd: ", size(Wd[1]));
	println("size HINV: ", size(HinvPs[1]));
	for i = 1:numOfCurrentProblems
		lhs += (mean(Wd[i])^2) .* Z1' * HinvPs[i] * HinvPs[i]' * Z1;
	end

	lhs += alpha * I;

	Z2 = lhs\rhs;
end

function freqContExtendedSources(mc, pInv::InverseParam, pMis::Array{RemoteChannel},
	nfreq::Int64, windowSize::Int64, Iact,
	mback::Union{Vector,AbstractFloat,AbstractModel, Array{Float64,1}},
	dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];

pFor = fetch(pMis[1]).pFor
mSizeMat = pFor.Mesh.n .+ 1;
m = mSizeMat[1];
n = mSizeMat[2];
mSizeVec = mSizeMat[1] * mSizeMat[2];
# Z = copy(pFor.originalSources);
# sizeQ = size(Z);
nrec = size(pFor.Receivers, 2);
sizeH = size(pFor.Ainv[1]);

nsrc = size(pFor.Sources, 2);
pFor = nothing
alpha = 2e-3;
p = 10;

for freqIdx = startFrom:nfreq
	Z1 = rand(ComplexF64,(m*n, p)) .+ 0.01;
	Z2 = rand(ComplexF64, (p, nsrc)) .+ 0.01;
	println("start freqCont Zs iteration from: ", freqIdx)
	tstart = time_ns();
	reqIdx1 = freqIdx;
	if freqIdx > 1
		reqIdx1 = max(1,freqIdx-windowSize+1);
	end
	reqIdx2 = freqIdx;
	currentProblems = reqIdx1:reqIdx2;


	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	runningProcs = map(x->x.where, pMis[currentProblems]);


	for j=1:5

		pMisCurrent = map(fetch, pMis[currentProblems]);
		pForpCurrent =  map(x->x.pFor, pMisCurrent);
		Dp,pForp = getData(vec(mc), pForpCurrent);

		pForCurrent = map(x->fetch(x), pForp);
		numOfCurrentProblems = size(currentProblems, 1);
		map((pm,pf) -> pm.pFor = pf , pMisCurrent, pForCurrent);
		HinvPs = Vector{Array}(undef, numOfCurrentProblems);
		pMisTemp = Array{RemoteChannel}(undef, length(currentProblems));
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

		t2 = time_ns();
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);
		e2 = time_ns();
		println("Misfit B4 mzs at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
		println("runtime of compute misfit");
		println((e2 - t2)/1.0e9);

		t3 = time_ns();

		pMisArr = map(pm -> fetch(pm), pMisCurrent);
		Ap = Vector{Matrix}(undef, numOfCurrentProblems);
		diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
		B = Vector{Vector}(undef, numOfCurrentProblems);

		toVec(mat) = reshape(mat, mSizeVec);
		toMat(vec) = reshape(vec, tuple(mSizeMat...));

		Wd = map(pm -> pm.Wd[:,:,1], pMisCurrent)
		function misfitCalc()
			sum = 0;
			for i = 1:numOfCurrentProblems
				sum += (mean(Wd[i])^2) .* norm(HinvPs[i]' * (pMisCurrent[i].pFor.Sources + Z1 * Z2) - pMisCurrent[i].dobs[:,:,1])^2;
			end

			sum	+= alpha * norm(Z1)^2 + alpha * norm(Z2)^2;
			return sum;
		end

		Z2 = calculateZ2(misfitCalc, p, nsrc, numOfCurrentProblems, Wd, HinvPs,
			pMisCurrent, Z1, alpha);

		println("misfit at Z2:: ", misfitCalc())

		function multOP(R,HinvP)
			return HinvP' * R * Z2;
		end

		function multOPT(R,HinvP)
			return HinvP * R * Z2';
		end

		function multAll(x)
			sum = zeros(ComplexF64, (mSizeVec, p));

			for i = 1:numOfCurrentProblems
				sum += (mean(Wd[i])^2) .* multOPT(multOP(x, HinvPs[i]), HinvPs[i]);
			end

			sum += alpha * x;
			return sum;
		end

		rhs = zeros(ComplexF64, (mSizeVec, p));
		for i = 1:numOfCurrentProblems
			pm = pMisCurrent[i]
			rhs += (mean(Wd[i])^2).*multOPT(-HinvPs[i]' * pm.pFor.Sources + pm.dobs[:,:,1], HinvPs[i]);
		end

		Z1 = KrylovMethods.blockBiCGSTB(x-> real(multAll(x)), real(rhs))[1];

		for i = 1:numOfCurrentProblems
			pMisCurrent[i].pFor.Sources += Z1 * Z2;
		end

		println("misfit at Z1:: ", misfitCalc())

		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);

		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
		pMispCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
		for i=1:numOfCurrentProblems
			pMispCurrent[i] = initRemoteChannel(x->x, runningProcs[i], pMisCurrent[i]);
		end
		pInv.mref = mc[:];
		t4 = time_ns();
		mc,Dc,flag,His = projGNCG(mc,pInv,pMispCurrent,dumpResults = dumpGN);
		e4 = time_ns();
		println((t4 - e4)/1.0e9);
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc,pMisCurrent);

		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);

		pMis[currentProblems] = pMispCurrent;
		clear!(pMispCurrent);
	end

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
	pMisTE = calculateReducedMisfitParams(mc, currentProblems, pMis, Iact, mback);
	pInv.mref = mc[:];

	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");

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

	mc,Dc,flag,His = projGNCG(mc,pInv,pMisTE,dumpResults = dumpGN);

	clear!(pMisTE);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end

return mc,Dc,flag,HIS;
end

"""
	Function to calculate MistfitParams of new dimensions after trace estimation
"""
function calculateReducedMisfitParams(mc, currentProblems::UnitRange, pMis::Array{RemoteChannel},
			Iact,mback::Union{Vector,AbstractFloat,AbstractModel})
	numOfCurrentProblems = size(currentProblems, 1);

	eta = 20.0;
	newDim = 10;

	runningProcs = map(x->x.where, pMis[currentProblems]);
	pMisCurrent = map(fetch, pMis[currentProblems]);
	pForpCurrent =  map(x->x.pFor, pMisCurrent);

	DobsCurrent = map(x->x.dobs[:,:], pMisCurrent);
	WdCurrent = map(x->x.Wd[:,:], pMisCurrent);
	DobsNew = copy(DobsCurrent[:]);
	nsrc = size(DobsNew[1],2);
	TEmat = rand([-1,1],(nsrc,newDim));
	WdEta = eta.*WdCurrent;
	DpCurrent, = getData(vec(mc),pForpCurrent);
	WdNew = Array{Array}(undef, numOfCurrentProblems);
	for i=1:numOfCurrentProblems
		DobsTemp = fetch(DpCurrent[i]);
		for s=1:nsrc
			etaM = 2 * eta .*diagm(0 => vec(WdCurrent[i][:,s])).*diagm(0 => vec(WdCurrent[i][:,s]));
			A = (2 .*I + etaM)
			b = (etaM)*(A\DobsCurrent[i][:,s])
			DobsNew[i][:,s] = 2 .* (A\DobsTemp[:,s]) + b;
		end
		DobsNew[i] = DobsNew[i][:,:] * TEmat;
		WdNew[i] = 1.0./(sqrt(newDim) .* abs.(real.(DobsNew[i])) .+ 1e-1*mean(abs.(DobsNew[i])));
	end

	pForReduced = Array{RemoteChannel}(undef, numOfCurrentProblems);
	for i=1:numOfCurrentProblems
		pForTemp = pForpCurrent[i];
		pForTemp.Sources = pForTemp.originalSources * TEmat;
		pForReduced[i] = initRemoteChannel(x->x, runningProcs[i], pForTemp);
	end
	return getMisfitParam(pForReduced, WdNew, DobsNew, SSDFun, Iact, mback);
end

function dummy(mc,Dc,iter,pInv,pMis)
# this function does nothing and is used as a default for dumpResults().
end
