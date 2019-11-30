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

function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, s::Int64, HinvPs::Vector{Array},
	reg::Vector,
	beta::Float64
	,
	writeSource::Function)
	recvSize = size(pMisArr[1].pFor.Receivers)

	# sizeZ = size(Z[:,s])
	println("starting: ", s);


	numOfCurrentProblems = size(currentProblems, 1);
	sumZs = 0;
	for i=1:numOfCurrentProblems
		# println(typeof(pMisArr))
		# println(typeof(pMisArr[i].Wd))
		# println("size wd: ", size(pMisArr[i].Wd[:,s]))
		WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
		# println("size pfor: ", size(pMisArr[i].pFor.Sources[:,s]))
		sumZs += norm(WdSqr * (HinvPs[i]' * pMisArr[i].pFor.Sources[:,s] -  pMisArr[i].dobs[:,s,1]));
	end
	# println("norm with Zs b4 cg:" , s, "is :", sumZs);
	# println("SIZE A;:", size(Ap))
	Ap = Vector{Matrix}(undef, numOfCurrentProblems);
	diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
	B = Vector{Vector}(undef, numOfCurrentProblems);
	# ref = Vector(pMisArr[1].pFor.Sources[:,s]) +  rand(sizeH[1]) * 100;
	ref = zeros(sizeH[1])
	# ref = rand(sizeH[1]) * 1000;
	# beta = 1e-8;
	# A = zeros(sizeH[1], sizeH[2]);
	# Bs = zeros(sizeH[1]);
	# PB = zeros(sizeH[1]);
	for i=1:numOfCurrentProblems
		P = pMisArr[i].pFor.Receivers;
		WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
		# LUcur = pMisArr[i].pFor.Ainv[1];
		Ap[i] = HinvPs[i]
		# LUcur = LUcur';
		# HinvP = zeros(ComplexF64, size(P))
		# HinvP = LUcur \ Matrix(P);
		# println("Size Hinv:", size(HinvP))
		# println(size(diag))
		# println(size(WdSqr))
		# Ap[i] =  real(HinvP);
		diags[i] =  WdSqr;
		# A1 = A + HinvP * WdSqr * HinvP'
		# A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta .* I;
		# PB += real(HinvP * WdSqr * pMisArr[i].dobs[:,s,1]);
		B[i] =  2 * beta.* pMisArr[i].pFor.originalSources[:, s] +  HinvPs[i] * WdSqr * pMisArr[i].dobs[:,s,1];
		# Bs = Bs + 2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[i].dobs[:,s,1];
	end

	# println("AABBB");
	# writedlm(string("B_",s), PB);
	# for i = 1:numOfCurrentProblems
	# 	# println("Size diags:", size(A))
	# 	# println("Size ap:", size(Ap[i]))
	# 	# println("Size dgs:", size(diags[i]))
	# 	A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta.* diagm(0 => vec(reg));
	# 	Bs = Bs + B[i]
	# end
# 	println("A11,", A[5,5])
# println("A11,", A[8,8])
# println("BB,", beta)
# println("BB1,", Bs[5])

	# Z[:, s] = A\B;
	# pForTemp = pForpCurrent[i];
	# pForTemp.Sources = Zs;
	# A = Ap * diag * Ap' + 2 * size(currentProblems) * beta.* diagm(0 => vec(reg)) * diagm(0 => vec(reg));
	# writedlm(string("B", s, "_", inx), Bs)
	# writedlm(string("Aq", s, "_", inx), A[:, :])
	# newSource = real(A\Vector(Bs));
		# H::Function,gc::Vector,Active::BitArray,Precond::Function,cgTol::Real,maxIter::Int;out::Int=0)
		# newSource = projPCG
		# A(x)=sum(map((Xp, diag) -> Xp * diag * (Xp' * x) .+ 2 * beta .* x, Ap, diags));
		# newSource,flagCG,relresCG,iterCG,resvecCG       = KrylovMethods.cg(A, sum(B), tol=1e-5, maxIter=100, out=2);
		# newSource = sum(map((Xp, diag) -> Xp * diag * Xp' , Ap, diags)) \ sum(B);
		# newSource = conjGradEx(beta, Ap, diags, B, sizeH[1], ref, 16);
	# println("SZ:",size(newSource))
	# newSource = jacobi(beta, sizeH, numOfCurrentProblems, pMisArr, s, 20);
	# println(nor)
	# A = zeros((sizeH[1], sizeH[1]));
	# b = zeros(sizeH[1]);
	# newSource = zeros(sizeH[1]);
	# for i=1:numOfCurrentProblems
	# 	WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
	# 	A += HinvPs[i] *WdSqr* HinvPs[i]' + 2*beta*I;
	# 	# println("ISPOSDEF FOR i ", i, " " , isposdef( HinvPs[i] * HinvPs[i]'))
	# 	# println(HinvPs[i] * HinvPs[i]');
	# 	b += HinvPs[i] * WdSqr * pMisArr[i].dobs[:,s,1] + 2 * beta .* Vector(pMisArr[i].pFor.originalSources[:, s])
	# 	# newSource += Vector(pMisArr[i].pFor.originalSources[:, s]) + (HinvPs[i] * WdSqr * pMisArr[i].dobs[:,s,1])./(2 * beta) -
	# 	# 	 ( HinvPs[i] *WdSqr* HinvPs[i]' * Vector(pMisArr[i].pFor.originalSources[:, s])) ./ (2*beta);
	# end
	# newSource = newSource ./ float(numOfCurrentProblems);
	# println("ISPOSDEF FOR A: " , isposdef(A))
	# println("type A ",typeof(A))
	# println("type b ",typeof(b))
	# println(A)
	# newSource = A\b;
	# A=real(A)
	# b=real(b)

	newSource = KrylovMethods.cg((x-> sum(map((Ai, diag) ->  Ai * diag * (Ai'* x) + 2*beta.*reg.* x, Ap, diags))), sum(B), tol=1e-5, maxIter=10)[1];
	writeSource(newSource, s);

	writeSource = nothing;
	# writedlm(string("Ns", s), newSource);
	# println(newSource);
	# println("type ns: ", typeof(newSource));

	sumZs = 0;
	sumQs = 0;
	sumZsD = 0;
	sumQsD = 0;
	for i=1:numOfCurrentProblems
		WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
		sumZs += norm(WdSqr * (HinvPs[i]' * newSource -  pMisArr[i].dobs[:,s,1])) + beta * norm(newSource - pMisArr[i].pFor.originalSources[:, s]);
		sumQs +=  norm(WdSqr * (HinvPs[i]' * pMisArr[i].pFor.originalSources[:, s] -  pMisArr[i].dobs[:,s,1]));
		sumZsD += norm((HinvPs[i] * HinvPs[i]' * newSource -  HinvPs[i] * pMisArr[i].dobs[:,s,1]));
		sumQsD +=  norm((HinvPs[i] * HinvPs[i]' * pMisArr[i].pFor.originalSources[:, s] - HinvPs[i] * pMisArr[i].dobs[:,s,1]));
	end

	# println("norm with Zs:" , s, "is :", sumZs);
	# println("norm with Qs:" , s, "is :", sumQs);
	#
	# println("norm with ZsD:" , s, "is :", sumZsD);
	# println("norm with QsD:" , s, "is :", sumQsD);


	# writedlm(string("Ns", s, "_", inx), newSource)
	return newSource
	# return pMisCurAll;
end
function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, indexes::StepRange,
	regByDist::Array,
	HinvPs::Vector{Array},
	beta::Float64
	, writeSource::Function)
	newSources = Array{Array}(undef, size(indexes))
	# println(inx);
	for (i, s) in enumerate(indexes)
		newSources[i] = calculateZs(currentProblems, sizeH, pMisArr, s, HinvPs,
		regByDist[s],
		beta
		, writeSource);
	end
	return newSources;
end

function minimizeZs(mc, currentProblems::UnitRange, HinvPs::Vector{Array},
	sizeH::Tuple, pMisArr::Array{MisfitParam}, nsrc::Integer, beta::Float64,
	regByDist::Array,
	writeSource::Function,
	runningProcs::Array)
	# pMisTemp = pMis[currentProblems];
	pMisTemp = Array{RemoteChannel}(undef, length(currentProblems));

	# println("ABCD3")

	# runningProcs = map(x->x.where, pMis[currentProblems]);

	# pForCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
	# Zs = Array{}
	t111 = time_ns();
	# pMisArr = Array{MisfitParam}(undef, length(currentProblems));
	# for i=1:length(currentProblems)
	# 	pMisArr[i] = fetch(pMisTemp[i])
	# end
	newSourcesp = Array{RemoteChannel}(undef, nworkers());
	@sync begin
		for worker in 1:nworkers()
			@async begin
				newSourcesp[worker] = initRemoteChannel(calculateZs, workers()[worker],
				currentProblems, sizeH, pMisArr, worker:nworkers():nsrc,
				regByDist,
				HinvPs, beta
				, writeSource);
			end
		end
	end
	newSources = Array{Array}(undef, nworkers());
	Qs = pMisArr[1].pFor.originalSources;
	s1 = size(Qs)[1];
	# println("size sources 1: ", s1);
	for worker in 1:nworkers()
		# newSource = calculateZs(currentProblems, sizeH, pMisArr, s);
		newSources[worker] = fetch(newSourcesp[worker])
		println("size for worker: ", size(newSources[worker]));
	end
	# println("TYPE OF NEW SOURCE ", typeof(newSources))
	# println("SIZE OF NEW SOURCE ", size(newSources))
	for i=1:size(currentProblems, 1)
		# pMisCur = fetch(pMisTemp[i]);
		# println("A/B");
		# println(size(newSource));
		Sources = Matrix{ComplexF64}(undef, s1,nsrc);
		# println("SIZE SOURCES : ", size(Sources));
		# println("type of sources: ", typeof(pMisArr[i].pFor.Sources))
		for s=1:nsrc
			# pMisArr[i].pFor.ExtendedSources[:, s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];
			# pMisArr[i].pFor.Sources[:, s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];
			Sources[:,s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];

			# writeSource(pMisArr[i].pFor.Sources[:, s], s);
		end
			# println("AFTER22");
		# writedlm(string("Src", i), pMisArr[i].pFor.Sources)
		pMisArr[i].pFor.Sources = Sources;
		pMisTemp[i] = initRemoteChannel(x->x, runningProcs[i], pMisArr[i]);
		# pMis[currentProblems] = pMisTemp;
	end

	s111 = time_ns();
	println("FREQCONT ZS");
	println((s111 - t111)/1.0e9);
	return pMisTemp;
end

function freqContZs(mc, pInv::InverseParam, pMis::Array{RemoteChannel}, srcLocations::Array{Int},
	nfreq::Int64, windowSize::Int64,
	Iact,mback::Union{Vector,AbstractFloat,AbstractModel, Array{Float64,1}},
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
# origGNIters = pInv.maxIter;
# firstRun = 12;
# pInv.maxIter = firstRun;
Dc = 0;
# println("GLOBAL INX");
flag = -1;
# println(inx)

HIS = [];

pFor = fetch(pMis[1]).pFor

Z = copy(pFor.originalSources);
nrec = size(pFor.Receivers, 2);
sizeH = size(pFor.Ainv[1]);

nsrc = size(Z, 2);


srcIndexes = map(loc -> [loc, 1], srcLocations);
meshLineSize = size(pFor.Receivers, 1);
regByDist = Array{Vector}(undef, nsrc);

meshSize = pFor.Mesh.n .+ 1;
for j = 1:nsrc
	regByDist[j] = zeros(meshLineSize);
	xs = srcIndexes[j];
	for i = 1:meshLineSize
		ind = [((i-1) % meshSize[1]) + 1, floor((i-1)/meshSize[1]) + 1];
		regByDist[j][i] = norm(xs - ind, 1) + 1;
	end
end

# println("SIZE Z");
# println(size(Z));
# println("NSRC221");
# println(nsrc);
pFor = nothing
beta = 1e-5;
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
	# pFor =  fetch(pMis[freqIdx]).pFor;

	itersNum = 5
	if freqIdx == startFrom
		itersNum = 10
	end

	for j=1:itersNum
		pMisCurrent = map(fetch, pMis[currentProblems]);
		pForpCurrent =  map(x->x.pFor, pMisCurrent);
		Dp,pForp = getData(vec(mc), pForpCurrent);
		# for i = 1:length(currentProblems)
		# 	pMisCurrent[i].dobs = fetch(Dp[i]);
		# end
		pForCurrent = map(x->fetch(x), pForp);
		numOfCurrentProblems = size(currentProblems, 1);
		# DobsC = Array{Array}(undef, numOfCurrentProblems);
		# for k=1:length(Dp)
		# 	DobsC[k] = fetch(Dp[k]);
		# end
		# map((pm, dobsCur) -> pm.dobs = dobsCur , pMisCurrent, DobsC);
		# WdC = Array{Array}(undef, numOfCurrentProblems);
		# for k=1:length(DobsC)
		# 	# Wd[k] = 1.0./(abs.(real.(Dobs[k])) .+ 1e-1*mean(abs.(Dobs[k])));
		# 	WdC[k] = ones(size(DobsC[k]))./(mean(abs.(DobsC[k])));
		# end
		map((pm,pf) -> pm.pFor = pf , pMisCurrent, pForCurrent);

		# fetchedPMis = Vector{MisfitParam}(undef, numOfCurrentProblems);
		HinvPs = Vector{Array}(undef, numOfCurrentProblems);

		for freqs = 1:numOfCurrentProblems
			# index = freqs - startFrom + 1;
			# fetchedPMis[freqs] = fetch(pMis[freqs]);
			HinvPs[freqs] = real((pForCurrent[freqs].Ainv[1])' \ Matrix(pForCurrent[freqs].Receivers));

			println("HINVP done");
			# pMisTemp = getMisfitParam(pForCurrent, WdNew, pMis., SSDFun, Iact, mback);
			# Zs =
			# println("SOURCES, ", fetch(pMisTemp[1]).pFor.Sources)
			# Here we set a dump function for GN for this iteracion of FC
		end

		# sumMs = 0;

		# for ix1 = startFrom:freqIdx
		# 	println(nsrc);
		# 	for s = 1:nsrc
		# 		s=1;
		# 		pMisCC = fetchedPMis[ix1];
		# 		WdSqr = 2 .*diagm(0 => vec(pMisCC.Wd[:,s])).*diagm(0 => vec(pMisCC.Wd[:,s]));
		# 		sumMs +=  norm(WdSqr * (HinvPs[ix1]' * pMisCC.pFor.originalSources[:, s] -  pMisCC.dobs[:,s,1]));
		# 	end
		# end
		# println("norm for M at iter: ", j, "WITH EXSRC is: ", sumMs);
		if resultsFilename == ""
				filename = "";
			else
				Temp = splitext(resultsFilename);
				filename = string(Temp[1],"_FC",freqIdx,"_",j,"_GN",Temp[2]);
		end

		function dumpGN(mc,Dc,iter,pInv,PF)
			dumpFun(mc,Dc,iter,pInv,PF,filename);
		end

		function writeSource(source, sourceIdx)
			if sourceIdx < 10
				writedlm(string("Ns", sourceIdx, "_GN", j, "_FC", freqIdx), source)
			end
		end

		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);
		println("Misfit B4 mzs at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);

		pMisTemp = minimizeZs(mc, currentProblems, HinvPs,sizeH, pMisCurrent, nsrc, beta,
		regByDist,
		writeSource,
		map(x->x.where, pMis[currentProblems]));




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
		mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);
		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, map(pm -> fetch(pm), pMisTemp));

		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);

		pMis[currentProblems] = pMisTemp;
		clear!(pMisTemp);
	end
	# if freqIdx > startFrom
		# beta *= 10;
	# end
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
	global inx = inx + 1;
	# pInv.maxIter = origGNIters;

end
# for i in 1:nfreq
# 	writedlm(string("sources", i), fetch(pMis[i]).pFor.Sources)
# end
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
