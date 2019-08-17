export freqContBasic;
export freqContTraceEstimation;
using jInv.InverseSolve
using Printf
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

	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
end
return mc,Dc,flag,HIS;
end

"""
Ap, diag are matrices such that diag is diagonal and ASum(Ap(i)*diag(i)*Ap(i)^T)
calculates x where Ax=b, relevant when A is too big to save in memory
"""
function conjGradEx(beta::Float64, Ap::Vector{Matrix}, diags::Vector{SparseMatrixCSC},
	b::Vector{Vector}, meshSize::Int, ref::Vector,
	numIters::Int)
	problemsSize = length(Ap);
	println("S", problemsSize);
	r = zeros(meshSize);

	for i = 1:problemsSize
		println(size(r))
		println(size(Ap[i]),size(diags[i]), size(ref))
		r += b[i] - Ap[i] * diags[i] * ((Ap[i])' * ref) - 2 * beta .* ref;
	end
	p = copy(r);
	# A = sum(map((Xp, diag) -> Xp * diag * Xp', Ap, diags));
	# println("SPD:", A' == A)
	B = sum(b);

	threshold = 1e-5;
	for i = 1:numIters
		println("Current Iter: ", i);
		# println("size ref", size(ref), size(A));
		# println("norm r ", norm(B - A*ref));
		divider = sum(map((Xp, diag) -> p' * Xp * diag * (Xp' * p) .+ 2 * beta .* p' * p, Ap, diags));
		alpha = (r' * r) / divider;
		println(alpha);
		refDiff = alpha .* p;
		# println("BBB")
		ref += refDiff;
		rNext = r -  alpha .* sum(map((Xp, diag) -> Xp * diag * (Xp' * p) + 2 * beta .* p, Ap, diags));
# println("BBB2")
		if norm(rNext) < threshold
			return ref;
		end
		# println("BBB3")
		bet = (rNext' * rNext) / (r' * r);
		# println("BBB4")
		r = rNext;
		p = r + bet .* p;
		println("refdiff:",norm(refDiff));
		println(norm(p));
	end

	return ref;
	# println(size(r0));

end

function jacobi(beta::Float64, sizeH::Tuple, numOfCurrentProblems::Int64,
	pMisArr::Array{MisfitParam}, s::Int64,
	numIters::Int)
	# problemsSize = length(Ap);

	Ap = Vector{Matrix}(undef, numOfCurrentProblems);
	diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
	B = Vector{Vector}(undef, numOfCurrentProblems);

	for j=1:numOfCurrentProblems
		P = pMisArr[j].pFor.Receivers;
		WdSqr = 2 .*diagm(0 => vec(pMisArr[j].Wd[:,s])).*diagm(0 => vec(pMisArr[j].Wd[:,s]));
		LUcur = pMisArr[j].pFor.Ainv[1];
		LUcur = LUcur';
		HinvP = zeros(ComplexF64, size(P))
		HinvP = LUcur \ Matrix(P);
		println("Size Hinv:", size(HinvP))
		# println(size(diag))
		# println(size(WdSqr))

		Ap[j] =  real(HinvP);
		diags[j] =  real(WdSqr);
		B[j] = real(2 * beta .* pMisArr[j].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[j].dobs[:,s,1]);

		# A1 = A + HinvP * WdSqr * HinvP'
		# A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta .* I;
		# b =  real(2 * beta .* pMisArr[j].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[j].dobs[:,s,1]);
		# r += b - Ap * WdSqr * (Ap' * ref) - 2 * beta .* ref;
		# D += map(x->x^2, Ap) * diag(WdSqr) .+ 2*beta;
		# Bs = Bs + 2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[i].dobs[:,s,1];
	end
	# ref = Vector(pMisArr[1].pFor.Sources[:,s])
	ref = zeros(sizeH[1])
	# beta = 1e-2;
	# A = zeros(sizeH[1], sizeH[2]);
	# Bs = zeros(sizeH[1]);
	threshold = 1e-4;
	meshSize = sizeH[1];
	for i = 1:numIters
		println("Current Iter: ", i);
		# println("size ref", size(ref), size(A));
		# println("norm r ", norm(B - A*ref));
		r = zeros(meshSize);
		D = zeros(meshSize);

		# for j = 1:problemsSize
		# 	# println(size(r))
		# 	# println(size(Ap[i]),size(diags[i]), size(ref))

		# 	# for k=1:meshSize
		# 	# 	sqrAp = map(x->x^2, Ap[j][k,:])
		# 	# 	D[k] = sqrAp * diag(diags[j])
		# 	# end
		# end
		for j=1:numOfCurrentProblems
			# P = pMisArr[j].pFor.Receivers;
			# WdSqr = 2 .*diagm(0 => vec(pMisArr[j].Wd[:,s])).*diagm(0 => vec(pMisArr[j].Wd[:,s]));
			# LUcur = pMisArr[j].pFor.Ainv[1];
			# LUcur = LUcur';
			# HinvP = zeros(ComplexF64, size(P))
			# HinvP = LUcur \ Matrix(P);
			# println("Size Hinv:", size(HinvP))
			# # println(size(diag))
			# # println(size(WdSqr))
			# Ap =  real(HinvP);
			# WdSqr =  real(WdSqr);
			# A1 = A + HinvP * WdSqr * HinvP'
			# A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta .* I;
			# b =  real(2 * beta .* pMisArr[j].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[j].dobs[:,s,1]);
			r += B[j] - Ap[j] * diags[j] * (Ap[j]' * ref) - 2 * beta .* ref;
			D += map(x->x^2, Ap[j]) * diag(diags[j]) .+ 2*beta;
			# Bs = Bs + 2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[i].dobs[:,s,1];
		end

		if norm(r) < threshold
			return ref;
		end
		# println("r:", r)
		# println("D:", D)
		# println("D-1:",map(x-> 1/x, D))
		# println(ref)
		ref += map(x-> 1/x, D) .* r;
	end
	# println("S", problemsSize);
	# p = copy(r);
	# A = sum(map((Xp, diag) -> Xp * diag * Xp', Ap, diags));
	# println("Anorm:", norm(Ao-A))
	# println("BDIFF:", norm(Bo-sum(b)))
	# println("ADIFF:", norm(Ao - sum(map((Xp, diag) -> Xp * diag * Xp' + 2 * beta .* I, Ap, diags))))
	# println("SPD:", A' == A)
	# B = sum(b);
	# divider = sum(map((Xp, diag) -> p' * Xp * diag * (Xp' * p), Ap, diags));
	# for i = 1:numIters
	# 	println("Current Iter: ", i);
	# 	# println("size ref", size(ref), size(A));
	# 	# println("norm r ", norm(B - A*ref));
	# 	r = zeros(meshSize);
	# 	D = zeros(meshSize);
	#
	# 	for j = 1:problemsSize
	# 	# 	# println(size(r))
	# 	# 	# println(size(Ap[i]),size(diags[i]), size(ref))
	# 		r += b[j] - Ap[j] * diags[j] * ((Ap[j])' * ref) - 2 * beta .* ref;
	# 		D += map(x->x^2, Ap[j]) * diag(diags[j]) .+ 2*beta;
	# 	# 	# for k=1:meshSize
	# 	# 	# 	sqrAp = map(x->x^2, Ap[j][k,:])
	# 	# 	# 	D[k] = sqrAp * diag(diags[j])
	# 	# 	# end
	# 	end
	# 	# println("SZ:",size(A),size(B))
	# 	# rt = Bo - Ao * ref
	# 	# println("DiffR:", norm(rt-r))
	# 	# println("SR",size(r))
	# 	# Dt = diag(Ao)
	# 	# println("DiffD:", norm(Dt-D))
	# 	if norm(r) < threshold
	# 		return ref;
	# 	end
	# 	# println("r:", r)
	# 	# println("D:", D)
	# 	# println("D-1:",map(x-> 1/x, D))
	# 	# println(ref)
	# 	ref += map(x-> 1/x, D) .* r;
	# 	# println(ref)
	# 	# alpha = (r' * r) / divider;
	# 	# println(alpha);
	# 	# refDiff = alpha * p;
	# 	# if norm(refDiff) < threshold
	# 	# 	return ref + refDiff;
	# 	# end
	# 	# ref += refDiff;
	# 	# rNext = r -  alpha * sum(map((Xp, diag) -> Xp * diag * (Xp' * p), Ap, diags));
	# 	# bet = (rNext' * rNext) / (r' * r);
	# 	# r = rNext;
	# 	# p = r + bet * p;
	# 	# println("refdiff:",norm(refDiff));
	# 	# println(norm(p));
	# end
	#
	# return ref;
	# # println(size(r0));
	return ref;
end

function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, s::Int64, HinvPs::Vector{Array}, beta::Float64)
	recvSize = size(pMisArr[1].pFor.Receivers)

	# sizeZ = size(Z[:,s])
	println("staring")
	numOfCurrentProblems = size(currentProblems, 1);
	# println("SIZE A;:", size(Ap))
	Ap = Vector{Matrix}(undef, numOfCurrentProblems);
	diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
	B = Vector{Vector}(undef, numOfCurrentProblems);
	# ref = Vector(pMisArr[1].pFor.Sources[:,s])
	ref = zeros(sizeH[1])
	# beta = 1e-8;
	# # A = zeros(sizeH[1], sizeH[2]);
	# # Bs = zeros(sizeH[1]);
	# PB = zeros(sizeH[1]);
	for i=1:numOfCurrentProblems
		P = pMisArr[i].pFor.Receivers;
		WdSqr = 2 .*diagm(0 => vec(pMisArr[i].Wd[:,s])).*diagm(0 => vec(pMisArr[i].Wd[:,s]));
		# LUcur = pMisArr[i].pFor.Ainv[1];
		Ap[i] = real(HinvPs[i])
		# LUcur = LUcur';
		# HinvP = zeros(ComplexF64, size(P))
		# HinvP = LUcur \ Matrix(P);
		# println("Size Hinv:", size(HinvP))
		# println(size(diag))
		# println(size(WdSqr))
		# Ap[i] =  real(HinvP);
		diags[i] =  real(WdSqr);
		# A1 = A + HinvP * WdSqr * HinvP'
		# A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta .* I;
		# PB += real(HinvP * WdSqr * pMisArr[i].dobs[:,s,1]);
		B[i] =  real(2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvPs[i] * WdSqr * pMisArr[i].dobs[:,s,1]);
		# Bs = Bs + 2 * beta .* pMisArr[i].pFor.originalSources[:, s] +  HinvP * WdSqr * pMisArr[i].dobs[:,s,1];
	end
	println("AABBB");
	# writedlm(string("B_",s), PB);
	# for i = 1:numOfCurrentProblems
	# 	# println("Size diags:", size(A))
	# 	# println("Size ap:", size(Ap[i]))
	# 	# println("Size dgs:", size(diags[i]))
	# 	A = A + Ap[i] * diags[i] * (Ap[i])' + 2 * beta .* I;
	# 	Bs = Bs + B[i]
	# end
# 	println("A11,", A[5,5])
# println("A11,", A[8,8])
# println("BB,", beta)
# println("BB1,", Bs[5])

	# Z[:, s] = A\B;
	# pForTemp = pForpCurrent[i];
	# pForTemp.Sources = Zs;
	# A = Ap * diag * Ap' + 2 * size(currentProblems) * beta.* I;
	# writedlm(string("B", s, "_", inx), Bs)
	# writedlm(string("Aq", s, "_", inx), A[:, :])
	# newSource = real(A\Vector(Bs));
	newSource = conjGradEx(beta, Ap, diags, B, sizeH[1], ref, 6);
	println("SZ:",size(newSource))
	# newSource = jacobi(beta, sizeH, numOfCurrentProblems, pMisArr, s, 20);
	# println(nor)
	writedlm(string("Ns", s, "_", inx), newSource)
	return newSource
	# return pMisCurAll;
end
function calculateZs(currentProblems::UnitRange, sizeH::Tuple,
	pMisArr::Array{MisfitParam}, indexes::StepRange, HinvPs::Vector{Array},
	beta::Float64)
	newSources = Array{Array}(undef, size(indexes))
	println(inx);
	for (i, s) in enumerate(indexes)
		newSources[i] = calculateZs(currentProblems, sizeH, pMisArr, s, HinvPs,
		beta);
	end
	return newSources;
end

function minimizeZs(mc, currentProblems::UnitRange, HinvPs::Vector{Array},
	sizeH::Tuple, pMis::Array{RemoteChannel}, nsrc::Integer, beta::Float64)
	pMisTemp = pMis[currentProblems];
	println("ABCD3")

	runningProcs = map(x->x.where, pMis[currentProblems]);

	# pForCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
	# Zs = Array{}
	t111 = time_ns();
	pMisArr = Array{MisfitParam}(undef, length(currentProblems));
	for i=1:length(currentProblems)
		pMisArr[i] = fetch(pMisTemp[i])
	end
	newSourcesp = Array{RemoteChannel}(undef, nworkers());
	@sync begin
		for worker in 1:nworkers()
			@async begin
				newSourcesp[worker] = initRemoteChannel(calculateZs, workers()[worker],
				currentProblems, sizeH, pMisArr, worker:nworkers():nsrc,
				HinvPs[currentProblems], beta);
			end
		end
	end
	newSources = Array{Array}(undef, nworkers());
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
		for s=1:nsrc
			pMisArr[i].pFor.Sources[:, s] = newSources[(s - 1) % nworkers() + 1][floor(Int64, (s - 1) / nworkers()) + 1];
		end
			# println("AFTER22");
		writedlm(string("Src", i), pMisArr[i].pFor.Sources)
		pMisTemp[i] = initRemoteChannel(x->x, runningProcs[i], pMisArr[i]);
		# pMis[currentProblems] = pMisTemp;
	end

	s111 = time_ns();
	println("FREQCONT ZS");
	println((s111 - t111)/1.0e9);
	return pMisTemp;
end

function freqContZs(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
	Iact,mback::Union{Vector,AbstractFloat,AbstractModel, Array{Float64,1}},
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
# origGNIters = pInv.maxIter;
# firstRun = 12;
# pInv.maxIter = firstRun;
Dc = 0;
println("GLOBAL INX");
flag = -1;
println(inx)
HIS = [];
pFor = fetch(pMis[1]).pFor
Z = copy(pFor.originalSources);
nrec = size(pFor.Receivers, 2);
sizeH = size(pFor.Ainv[1]);
pFor = nothing
nsrc = size(Z, 2);
println("SIZE Z");
println(size(Z));
println("NSRC221");
println(nsrc);
HinvPs = Vector{Array}(undef, length(startFrom:nfreq));
beta = 1e-8;
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
	pFor =  fetch(pMis[freqIdx]).pFor;
	HinvPs[freqIdx] = (pFor.Ainv[1])' \ Matrix(pFor.Receivers);

	# pMisTemp = getMisfitParam(pForCurrent, WdNew, pMis., SSDFun, Iact, mback);
	# Zs =
	# println("SOURCES, ", fetch(pMisTemp[1]).pFor.Sources)
	# Here we set a dump function for GN for this iteracion of FC
	pInv.mref = mc[:];

	for j=1:5
		if resultsFilename == ""
				filename = "";
			else
				Temp = splitext(resultsFilename);
				filename = string(Temp[1],"_FC",freqIdx,"_",j,"_GN",Temp[2]);
		end

		function dumpGN(mc,Dc,iter,pInv,PF)
			dumpFun(mc,Dc,iter,pInv,PF,filename);
		end

		pMisTemp = minimizeZs(mc, currentProblems, HinvPs,sizeH, pMis, nsrc, beta);

		mc,Dc,flag,His = projGN(mc,pInv,pMisTemp,dumpResults = dumpGN);


		pMis[currentProblems] = pMisTemp;
		clear!(pMisTemp);
	end
	beta *= 10;
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

function  projGNZs(mc,pInv::InverseParam,pMis;indCredit=[],
	dumpResults::Function = dummy,out::Int=2,solveGN::Function=projPCG)
	println("CHANGES PROJGN SAGIBU");

	maxIter     = pInv.maxIter      #  Max. no. iterations.
	pcgMaxIter  = pInv.pcgMaxIter   #  Max cg iters.
	pcgTol      = pInv.pcgTol       #  CG stopping tolerance.
	stepTol     = pInv.minUpdate    #  Step norm stopping tol.
	maxStep     = pInv.maxStep
	low         = pInv.boundsLow
	high        = pInv.boundsHigh
	alpha       = pInv.alpha

	His = getGNhis(maxIter,pcgMaxIter)
	#---------------------------------------------------------------------------
	#  Initialization.
	#---------------------------------------------------------------------------

	Active = (mc .<=low) .| (mc.>=high)  # Compute active set


	## evaluate function and derivatives
	sig,dsig = pInv.modelfun(mc)
	if isempty(indCredit)
		Dc,F,dF,d2F,pMis,tMis = computeMisfitZs(sig,pMis,true)
	else
		Dc,F,dF,d2F,pMis,tMis,indDebit = computeMisfitZs(sig,pMis,true,indCredit=indCredit)
	end
	dF = dsig'*dF
	dumpResults(mc,Dc,0,pInv,pMis) # dump initial model and dpred0

	# compute regularizer

	tReg = @elapsed R,dR,d2R = computeRegularizer(pInv.regularizer,mc,pInv.mref,pInv.MInv,alpha)

	# objective function
	Jc  = F  + R
	gc  = dF + dR

	F0 = F; J0 = Jc
	############################################################################
	##  Outer iteration.                                                        #
	############################################################################
	iter = 0
	outerFlag = -1; stepNorm=0.0

	outStr = @sprintf("\n %4s\t%08s\t%08s\t%08s\t%08s\t%08s\n",
					  	"i.LS", "F", "R","alpha[1]","Jc/J0","#Active")
	updateHis!(0,His,Jc,norm(projGrad(gc,mc,low,high)),F,Dc,R,alpha[1],count(!iszero, Active),0.0,-1,tMis,tReg)

	if out>=2; print(outStr); end
	f = open("jInv.out", "w")
	write(f, outStr)
	close(f)

	while outerFlag == -1

		iter += 1
		outStr = @sprintf("%3d.0\t%3.2e\t%3.2e\t%3.2e\t%3.2e\t%3d\n",
		         iter, F, R,alpha[1],Jc/J0,count(!iszero, Active))
		if out>=2; print(outStr); end
		f = open("jInv.out", "a")
		write(f, outStr)
		close(f)

		# solve linear system to find search direction
		His.timeLinSol[iter+1] += @elapsed  delm,hisLinSol = solveGN(gc,pMis,pInv,sig,dsig,d2F,d2R,Active)
		push!(His.hisLinSol,hisLinSol)

		# scale step
		if maximum(abs.(delm)) > maxStep; delm = delm./maximum(abs.(delm))*maxStep; end

		# take gradient direction in the active cells
		ga = -gc[mc .== low]
		if !isempty(ga)
			if maximum(abs.(ga)) > maximum(abs.(delm)); ga = ga./maximum(abs.(ga))*maximum(abs.(delm)); end
			delm[mc .== low] = ga
		end
		ga = -gc[mc .== high]
		if !isempty(ga)
			if maximum(abs.(ga)) > maximum(abs.(delm)); ga = ga./maximum(abs.(ga))*maximum(abs.(delm)); end
			delm[mc .== high] = ga
		end

		## Begin projected Armijo line search
		muLS = 1; lsIter = 1; mt = zeros(size(mc)); Jt = Jc
		while true
			mt = mc + muLS*delm
			mt[mt.<low]  = low[mt.<low]
			mt[mt.>high] = high[mt.>high]

			His.timeReg[iter+1] += @elapsed R,dR,d2R = computeRegularizer(pInv.regularizer,mt,pInv.mref,pInv.MInv,alpha)

			if R!=Inf # to support barrier functions.
				## evaluate function

				sigt, = pInv.modelfun(mt)
				if isempty(indCredit)
					Dc,F,dF,d2F,pMis,tMis = computeMisfitZs(sigt,pMis,false)
				else
					Dc,F,dF,d2F,pMis,tMis,indDebit = computeMisfitZs(sigt,false,indCredit=indCredit)
				end
				His.timeMisfit[iter+1,:]+=tMis


				# objective function
				Jt  = F  + R
				if out>=2;
					println(@sprintf( "   .%d\t%3.2e\t%3.2e\t\t\t%3.2e",
						lsIter, F,       R,       Jt/J0))
				end

				if Jt < Jc
					break
				end
			else
				Jt  = R;
				F = 0.0;
				if out>=2;
					println(@sprintf( "   .%d\t%3.2e\t%3.2e\t\t\t%3.2e",
						lsIter, F,       R,       Jt/J0))
				end
			end
			muLS /=2; lsIter += 1
			if lsIter > 6
			    outerFlag = -2
				break
			end
		end
		## End Line search

		## Check for termination
		stepNorm = norm(mt-mc,Inf)
		mc = mt
		Jc = Jt

		sig, dsig = pInv.modelfun(mc)

		Active = (mc .<=low) .| (mc.>=high)  # Compute active set

		#  Check stopping criteria for outer iteration.
		updateHis!(iter,His,Jc,-1.0,F,Dc,R,alpha[1],count(!iszero, Active),stepNorm,lsIter,tMis,tReg)

		dumpResults(mc,Dc,iter,pInv,pMis);
		if stepNorm < stepTol
			outerFlag = 1
			break
		elseif iter >= maxIter
			break
		end
		# Evaluate gradient

		if isempty(indCredit)
			His.timeGradMisfit[iter+1]+= @elapsed dF = computeGradMisfit(sig,Dc,pMis)
		else
			His.timeGradMisfit[iter+1]+= @elapsed dF = computeGradMisfit(sig,Dcp,pMis,indDebit)
		end

		dF = dsig'*dF
		gc = dF + dR


		His.dJ[iter+1] = norm(projGrad(gc,mc,low,high))

	end # while outer_flag == 0

	if out>=1
		if outerFlag==-1
			println("projGN iterated maxIter=$maxIter times but reached only stepNorm of $(stepNorm) instead $(stepTol)." )
		elseif outerFlag==-2
			println("projGN stopped at iteration $iter with a line search fail.")
		elseif outerFlag==1
			println("projGN reached desired accuracy at iteration $iter.")
		end
	end

	return mc,Dc,outerFlag,His
end  # Optimization code

function computeMisfitZs(sigmaRef::RemoteChannel,
                        pMisRef::RemoteChannel,
				      dFRef::RemoteChannel,
                  doDerivative,doClear::Bool=false)
#=
 computeMisfit for single forward problem

 Note: model (including interpolation matrix) and forward problems are RemoteRefs
=#

    rrlocs = [ pMisRef.where  dFRef.where]
    if !all(rrlocs .== myid())
        warn("computeMisfit: Problem on worker $(myid()) not all remote refs are stored here, but rrlocs=$rrlocs")
    end

    sigma = fetch(sigmaRef)
    pMis  = take!(pMisRef)

    Dc,F,dFi,d2F,pMis,times = computeMisfitZs(sigma,pMis,doDerivative,doClear)

    put!(pMisRef,pMis)
    # add to gradient
    if doDerivative
        dF = take!(dFRef)
        put!(dFRef,dF += dFi)
    end
    # put predicted data and d2F into remote refs (no need to communicate them)
    Dc  = remotecall(identity,myid(),Dc)
    d2F = remotecall(identity,myid(),d2F)

    return Dc,F,d2F,times
end


function computeMisfitZs(sigma,
	pMisRefs::Array{RemoteChannel,1},
	doDerivative::Bool=true;
	indCredit::AbstractVector=1:length(pMisRefs),
    printProgress::Bool=false)
#=
computeMisfit for multiple forward problems

This method runs in parallel (iff nworkers()> 1 )

Note: ForwardProblems and Mesh-2-Mesh Interpolation are RemoteRefs
    (i.e. they are stored in memory of a particular worker).
=#

    n = 1

	F   = 0.0
	dF  = (doDerivative) ? zeros(length(sigma)) : []
	d2F = Array{Any}(undef, length(pMisRefs));
	Dc  = Array{Future}(undef,size(pMisRefs))

	indDebit = []
	updateRes(Fi,idx) = (F+=Fi;push!(indDebit,idx))
	updateDF(x) = (dF+=x)

    workerList = []
    for k=indCredit
        push!(workerList,pMisRefs[k].where)
    end
    workerList = unique(workerList)
    sigRef = Array{RemoteChannel}(undef,maximum(workers()))
	dFiRef = Array{RemoteChannel}(undef,maximum(workers()))

	times = zeros(4);
	updateTimes(tt) = (times+=tt)

	@sync begin
		for p=workerList
			@async begin
				# communicate model and allocate RemoteRef for gradient
				sigRef[p] = initRemoteChannel(identity,p,sigma)   # send conductivity to workers
				dFiRef[p] = initRemoteChannel(zeros,p,length(sigma)) # get remote Ref to part of gradient
				# solve forward problems
				for idx in indCredit
					if pMisRefs[idx].where==p
						Dc[idx],Fi,d2F[idx],tt = remotecall_fetch(computeMisfitZs,p,sigRef[p],pMisRefs[idx],dFiRef[p],doDerivative)
						updateRes(Fi,idx)
						updateTimes(tt)
                        if printProgress && ((length(indDebit)/length(indCredit)) > n*0.1)
                            if doDerivative
                                println("Misfit and gradients computed for $(10*n)% of forward problems")
                            else
                                println("Misfit and gradients computed for $(10*n)% of forward problems")
                            end
                            n += 1
                        end
					end
				end

				# sum up gradients
				if doDerivative
					updateDF(fetch(dFiRef[p]))
				end
			end
		end
	end
	return Dc,F,dF,d2F,pMisRefs,times,indDebit
end

function computeMisfitZs(sig,
                       pMis::MisfitParam,doDerivative::Bool=true, doClear::Bool=false;
                       printProgress=false)
    if printProgress
        error("Print progress only works with multiple pFors")
    end
#=
 computeMisfit for a single forward problem. Everything is stored in memory on the node executing this function.
=#

    times = zeros(4)
    sigma,dsigma = pMis.modelfun(sig)
    times[1] = @elapsed   sigmaloc = interpGlobalToLocal(sigma,pMis.gloc.PForInv,pMis.gloc.sigmaBackground);
    times[2] = @elapsed   Dc,pMis.pFor  = getData(sigmaloc,pMis.pFor)      # fwd model to get predicted data
    times[3] = @elapsed   F,dF,d2F = pMis.misfit(Dc,pMis.dobs,pMis.Wd)
    if doDerivative
        times[4] = @elapsed dF = dsigma'*interpLocalToGlobal(getSensTMatVec(dF,sigmaloc,pMis.pFor),pMis.gloc.PForInv)
    end

    if doClear; clear!(pMis.pFor.Ainv); end
    return Dc,F,dF,d2F,pMis,times
end


function computeMisfitZs(sigmaRef::RemoteChannel,
                        pMisRef::RemoteChannel,
				      dFRef::RemoteChannel,
                  doDerivative,doClear::Bool=false)
#=
 computeMisfit for single forward problem

 Note: model (including interpolation matrix) and forward problems are RemoteRefs
=#

    rrlocs = [ pMisRef.where  dFRef.where]
    if !all(rrlocs .== myid())
        warn("computeMisfit: Problem on worker $(myid()) not all remote refs are stored here, but rrlocs=$rrlocs")
    end

    sigma = fetch(sigmaRef)
    pMis  = take!(pMisRef)

    Dc,F,dFi,d2F,pMis,times = computeMisfitZs(sigma,pMis,doDerivative,doClear)

    put!(pMisRef,pMis)
    # add to gradient
    if doDerivative
        dF = take!(dFRef)
        put!(dFRef,dF += dFi)
    end
    # put predicted data and d2F into remote refs (no need to communicate them)
    Dc  = remotecall(identity,myid(),Dc)
    d2F = remotecall(identity,myid(),d2F)

    return Dc,F,d2F,times
end


function computeMisfitZs(sigma,
	pMisRefs::Array{RemoteChannel,1},
	doDerivative::Bool=true;
	indCredit::AbstractVector=1:length(pMisRefs),
    printProgress::Bool=false)
#=
computeMisfit for multiple forward problems

This method runs in parallel (iff nworkers()> 1 )

Note: ForwardProblems and Mesh-2-Mesh Interpolation are RemoteRefs
    (i.e. they are stored in memory of a particular worker).
=#

    n = 1

	F   = 0.0
	dF  = (doDerivative) ? zeros(length(sigma)) : []
	d2F = Array{Any}(undef, length(pMisRefs));
	Dc  = Array{Future}(undef,size(pMisRefs))

	indDebit = []
	updateRes(Fi,idx) = (F+=Fi;push!(indDebit,idx))
	updateDF(x) = (dF+=x)

    workerList = []
    for k=indCredit
        push!(workerList,pMisRefs[k].where)
    end
    workerList = unique(workerList)
    sigRef = Array{RemoteChannel}(undef,maximum(workers()))
	dFiRef = Array{RemoteChannel}(undef,maximum(workers()))

	times = zeros(4);
	updateTimes(tt) = (times+=tt)

	@sync begin
		for p=workerList
			@async begin
				# communicate model and allocate RemoteRef for gradient
				sigRef[p] = initRemoteChannel(identity,p,sigma)   # send conductivity to workers
				dFiRef[p] = initRemoteChannel(zeros,p,length(sigma)) # get remote Ref to part of gradient
				# solve forward problems
				for idx in indCredit
					if pMisRefs[idx].where==p
						Dc[idx],Fi,d2F[idx],tt = remotecall_fetch(computeMisfitZs,p,sigRef[p],pMisRefs[idx],dFiRef[p],doDerivative)
						updateRes(Fi,idx)
						updateTimes(tt)
                        if printProgress && ((length(indDebit)/length(indCredit)) > n*0.1)
                            if doDerivative
                                println("Misfit and gradients computed for $(10*n)% of forward problems")
                            else
                                println("Misfit and gradients computed for $(10*n)% of forward problems")
                            end
                            n += 1
                        end
					end
				end

				# sum up gradients
				if doDerivative
					updateDF(fetch(dFiRef[p]))
				end
			end
		end
	end
	return Dc,F,dF,d2F,pMisRefs,times,indDebit
end


function computeMisfitZs(sigma,pMis::Array,doDerivative::Bool=true,indCredit=collect(1:length(pMis));
                       printProgress=false)
	#
	#	computeMisfit for multiple forward problems
	#
	#	This method runs in parallel (iff nworkers()> 1 )
	#
	#	Note: ForwardProblems and Mesh-2-Mesh Interpolation are stored on the main processor
	#		  and then sent to a particular worker, which returns an updated pFor.
	#
	numFor   = length(pMis)
 	F        = 0.0
    dF       = (doDerivative) ? zeros(length(sigma)) : []
 	d2F      = Array{Any}(undef,numFor)
 	Dc       = Array{Any}(undef,numFor)
	indDebit = []

	# draw next problem to be solved
	nextidx() = (idx = (isempty(indCredit)) ? -1 : pop!(indCredit))

 	updateRes(Fi,dFi,idx) = (F+=Fi; dF= (doDerivative) ? dF+dFi : []; push!(indDebit,idx))

	times = zeros(4);
	updateTimes(tt) = (times+=tt)

 	@sync begin
 		for p = workers()
 				@async begin
 					while true
 						idx = nextidx()
 						if idx == -1
 							break
 						end
 							Dc[idx],Fi,dFi,d2F[idx],pMis[idx],tt = remotecall_fetch(computeMisfitZs,p,sigma,pMis[idx],doDerivative)
 							updateRes(Fi,dFi,idx)
							updateTimes(tt)
 					end
 				end
 		end
 	end

 	return Dc,F,dF,d2F,pMis,times,indDebit
 end
