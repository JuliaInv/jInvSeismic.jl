export freqCont, freqContExtendedSources, freqContExtendedSourcesSS, freqCont;
using JLD
using Multigrid.ParallelJuliaSolver
using Statistics

export FreqContParam, getFreqContParams
"""
	FreqContParam

	parameters for the frequency continuation method

	Input:
		mc    		- current model
		itersNum	- number of GN iters for each frequency continuation step
		originalSources	- original sources before sources extension
		nrcv		- number of receivers
		pInv		- Inverse param
		pMis 		- misfit params (remote)
		windowSize  - How many frequencies to treat at once at the most.
		resultsFilename - a filename for saving the intermediate results according to the GN and continuation (FC) iterations (done in dumpFun())
		dumpFun     - a function for plotting, saving and doing all the things with the intermidiate results.
		Iact		- active set locations
		mback		- active set model values
		Z1			- first part of the extended sources initial guess
		simSrcDim	- dimensions of the simultaneous sources
		alpha1		- Z1 penalty coeficient
		alpha2Orig	- Z2 penalty coeficient
		stepReg		- regularization coeficient on the step size
		mode        - either "1stInit" or anything else.
					  1stInit will use the first group of misfits as an initialization and will not include it together with the next group of misfits.
		startFrom   - a start index for the continuation. Usefull when our run broke in the middle of the iterations.
		endAt		- index of the final frequency for the continuation.Usefull when we want to use only low frequencies.
		cycle       - just for identifying different runs when saving the results.
		method      - "projGN" or "barrierGN"
		FWImethod	- "FWI" or "FWI_ES"
		updateMref	- wheter to update the mref after each GN iteration, relevant for the ES case.

"""
mutable struct FreqContParam
	mc						:: Vector{Float64}
	itersNum				:: Int64
	originalSources			:: SparseMatrixCSC
	nrcv					:: Int64
	pInv					:: InverseParam
	pMis					:: Array{RemoteChannel}
	windowSize				:: Int64
	resultsFilename			:: String
	dumpFun					:: Function
	Iact					:: SparseMatrixCSC
	mback					:: Vector{Float64}
	Z1						:: Union{Int64, Array{ComplexF64, 2}}
	simSrcDim				:: Int64
	alpha1					:: Float64
	alpha2Orig				:: Float64
	stepReg					:: Float64
	mode					:: String
	startFrom				:: Int64
	endAt					:: Int64
	cycle					:: Int64
	method					:: String
	FWImethod				:: String
	updateMref				:: Bool
end

function getFreqContParams(mc,itersNum::Int64,
		originalSources::SparseMatrixCSC,nrcv, pInv::InverseParam,
		pMis::Array{RemoteChannel}, windowSize::Int64, resultsFilename::String,
		dumpFun::Function,Iact,mback;Z1=0,simSrcDim=1,
		alpha1=0.0, alpha2Orig=0.0, stepReg=0.0, mode::String="", startFrom::Int64 = 1,
		endAt=length(pMis),cycle::Int64=0, method::String="projGN",
		FWImethod::String="FWI",updateMref=false)
		return FreqContParam(mc, itersNum, originalSources,
				nrcv, pInv, pMis, windowSize, resultsFilename, dumpFun,
				Iact, mback, Z1, simSrcDim, alpha1, alpha2Orig, stepReg,
				mode, startFrom, endAt, cycle, method, FWImethod, updateMref)
end

function calculateZ2(m,misfitCalc::Function, p::Integer, nsrc::Integer, nfreq::Integer,
	nrec::Integer, nwork::Integer,
	numOfCurrentProblems::Integer, mergedWd::Array, mergedRc::Array, pMis::Array,
	pMisCurrent::Array{MisfitParam}, Z1::Matrix, alpha::Float64)

	rhs = zeros(ComplexF64, (p, nsrc));
	lhs = zeros(ComplexF64, (p,p));

	HinvPsZ1 = computeHinvTRecX(pMis, Z1, m, 0);

	function calculateLhs(meanWd,Z1,HinvP)
		return meanWd .* HinvP' * HinvP;
	end

	function calculateRhs(meanWd,HinvP,mergedRc)
		return meanWd .* HinvP' * mergedRc;
	end

	lhsides = Array{Array{ComplexF64}}(undef,nfreq)
	rhsides = Array{Array{ComplexF64}}(undef,nfreq)

	@sync begin
	 		for k=1:nfreq
		@async begin
			meanWd = mean(mergedWd[k])
			meanWdC = (real(meanWd).^2)
				lhsides[k] = remotecall_fetch(calculateLhs, k % nworkers() + 1,meanWdC,Z1,HinvPsZ1[k]);
				rhsides[k] = remotecall_fetch(calculateRhs, k % nworkers() + 1,meanWdC,HinvPsZ1[k], mergedRc[k]);
	 		end
	 	end
	end

	lhs = sum(lhsides)
	rhs = sum(rhsides)
	lhs += alpha * I;

	return lhs\rhs;
end

function MyPCG(A::Function,b::Array,x::Array,M::Array,numiter)
r = b-A(x);
z = M.*r;
p = copy(z);
norms = [norm(r)];

# use 20 iters if starting residual is high
if norms[1] > 100
	numiter=20;
end
for k=1:numiter
	Ap = A(p);
	alpha = real(dot(z,r)/dot(p,Ap));
	x .+= alpha*p;
	beta = real(dot(z,r));
	r .-= alpha*Ap
	z = M.*r
	nn = norm(r);
	if (nn < 0.1 * norms[1] || nn < 1e-2)
		break
	end
	norms = [norms; nn];
	beta = real(dot(z,r)) / beta;
	p = z + beta*p;
end
return x,norms
end

function misfitCalc(m, Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2,pMis)
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	sum = 0.0;
	HinvPs = computeHinvTRecX(pMis, Z1, m, 0);
	for i = 1:nfreq
		res, = SSDFun(HinvPs[i] * Z2,mergedRc[i],mean(mergedWd[i]) .* ones(size(mergedRc[i])));
		sum += res;
	end
	return sum;
end

function objectiveCalc(Z1,Z2,misfit,alpha1,alpha2)
	return misfit + 0.5*alpha1 * norm(Z1)^2 + 0.5*alpha2 * norm(Z2)^2;
end

function MultOp(HPinv, R, Z2)
	return (HPinv' * R) * Z2
end

function MultOpT(HPinv, R, Z2)
	return HPinv * (R * Z2')
end

function MultAll(m, avgWds, pMis, R, Z1, Z2, alpha, stepReg)
	sum = zeros(ComplexF64, size(R))
	inner = computeHinvTRecX(pMis, R,m, 0);

	function calculateInner(avgWd, inner, Z2)
		return avgWd^2 .* inner * (Z2 * Z2');
	end
	@sync begin
	 		for k=1:length(avgWds)
		@async begin
	 			inner[k] = remotecall_fetch(calculateInner,
	 				k % nworkers() + 1,avgWds[k],inner[k],Z2);
	 		end
	 	end
	end

	outer = computeHinvTRecXarr(pMis, inner,m, 1);
	for i = 1:length(avgWds)
		sum += outer[i]
	end
	eps = 1e-2
	return sum + (alpha./(abs.(Z1) .+ eps * maximum(abs.(Z1)))).*R + stepReg*R;
end

function calculateZ1(m, misfitCalc::Function, nfreq::Integer, mergedWd::Array, mergedRc::Array, pMis::Array, Z1::Matrix,Z2::Matrix, alpha1::Float64,stepReg::Float64)
	rhsElement = computeHinvTRecXarr(pMis, mergedRc, m, 1);
	rhsides = Array{Array{ComplexF64}}(undef,nfreq)


	function calculateRhs(meanWd,Z2,rhsElement)
		return meanWd.*  (rhsElement * Z2');
	end

	@sync begin
	 		for k=1:nfreq
		@async begin
			meanWd = mean(mergedWd[k])
			meanWdC = (real(meanWd).^2)
			rhsides[k] = remotecall_fetch(calculateRhs, k % nworkers() + 1,meanWdC,Z2, rhsElement[k]);
	 		end
	 	end
	end

	rhs = sum(rhsides) + (stepReg) .* Z1

	OP = x-> MultAll(m, mean.(real.(mergedWd)), pMis, x, Z1, Z2, alpha1, stepReg);
	eps = 10.0
	M = (abs.(Z1) .+ eps * maximum(abs.(Z1)));
	Z1, = MyPCG(OP,rhs,Z1,M,5);
	return Z1;
end

function standardGNrun(method, mc, simSrcDim, originalSources, pInv, pMisTemp,
		HIS, resultsFilename, cycle, freqIdx, dumpFun)
pMisTempFetched = map(fetch, pMisTemp);
dobs = map(x -> x.dobs,pMisTempFetched)
wd = map(x-> x.Wd,pMisTempFetched)
nsrc = size(originalSources,2);

if simSrcDim==1
	TEmat = Matrix(1.0I,nsrc,nsrc)
else
	TEmat = rand([-1,1],(nsrc,simSrcDim));
end

if simSrcDim > 1
	reducedSources = originalSources * TEmat;
	reducedDobs = map(x-> x*TEmat, dobs);
	sizeWD = size(reducedDobs[1])
	reducedWd = map(x-> mean(x)/sqrt(simSrcDim) * ones(sizeWD), wd);

	pMisTemp = setSourcesSame(pMisTemp,reducedSources);
	pMisTemp = setDobs(pMisTemp,reducedDobs);
	pMisTemp = setWd(pMisTemp,reducedWd);
end

if resultsFilename == ""
	filename = "";
	hisMatFileName = "";
else
	Temp = splitext(resultsFilename);
	if cycle==0
		filename = string(Temp[1],"_FC",freqIdx,"_GN",Temp[2]);
		hisMatFileName  = string(Temp[1],"_FC",freqIdx);
	else
		filename = string(Temp[1],"_Cyc",cycle,"_FC",freqIdx,"_GN",Temp[2]);
		hisMatFileName  =  string(Temp[1],"_Cyc",cycle,"_FC",freqIdx);
	end
end

# Here we set a dump function for GN for this iteracion of FC
function dumpGN(mc,Dc,iter,pInv,PF)
	dumpFun(mc,Dc,iter,pInv,PF,filename);
end

if method == "projGN"
	mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);
elseif method == "barrierGN"
	mc,Dc,flag,His = barrierGNCG(mc,pInv,pMisTemp,rho=1.0,dumpResults = dumpGN);
end

if simSrcDim > 1
	# set full data back
	pMisTemp = setSourcesSame(pMisTemp,originalSources);
	pMisTemp = setDobs(pMisTemp,dobs)
	pMisTemp = setWd(pMisTemp,wd)
end

if hisMatFileName != ""
	file = matopen(string(hisMatFileName,"_HisGN.mat"), "w");
	### PUT THE CONTENT OF "HIS" INTO THE MAT FILE ###
	His.Dc = [];
	write(file,"His",His);
	close(file);
end
His.Dc = []
push!(HIS,His)

clear!(pMisTemp);

return mc,Dc,flag,HIS
end

function ExtendedSourcesGNRun(method, mc, simSrcDim, nrcv, itersNum, originalSources,
		Z1, alpha1, alpha2, stepReg, pInv, pMisTemp, HIS, resultsFilename,
		cycle, freqIdx, dumpFun)
numOfCurrentProblems = length(pMisTemp);
N_nodes = prod(pInv.MInv.n .+ 1);
nsrc = size(originalSources, 2);
p = size(Z1,2);
nwork = nworkers()
nsrc = size(originalSources, 2);
nfreq = length(pMisTemp);
wheres = map(x -> x.where, pMisTemp);
pMisTempFetched = map(fetch, pMisTemp);
mergedDobs = map(x -> x.dobs,pMisTempFetched)
mergedWd = map(x-> x.Wd,pMisTempFetched)

##########################
# Initialize outputs	 #
##########################
Dc = 0;
flag = -1;
HIS = [];
Z2 = 0;
##########################

FafterGN = 0.0;
F_zero_prev = 0.0;
mc_prev = convert(Array{Float16},mc);
for j = 1:itersNum
	println("============================== New ALM Iter ======================================");
	flush(Base.stdout)

	if simSrcDim==1
		TEmat = Matrix(1.0I,nsrc,nsrc)
	else
		TEmat = rand([-1,1],(nsrc,simSrcDim));
	end

	reducedSources = originalSources * TEmat;
	reducedDobs = map(x-> x*TEmat, mergedDobs);
	sizeWD = size(reducedDobs[1])
	reducedWd = map(x-> mean(x)/sqrt(simSrcDim) * ones(sizeWD), mergedWd);

	pMisTemp = setSourcesSame(pMisTemp,reducedSources);
	pMisTemp = setDobs(pMisTemp,reducedDobs);
	pMisTemp = setWd(pMisTemp,reducedWd);

	t1 = time_ns();
	Dc,F_zero, = computeMisfit(mc,pMisTemp,false);
	e1 = time_ns();
	println("Computed Misfit with orig sources : ",F_zero, " [Time: ",(e1 - t1)/1.0e9," sec]");

	mc_prev = convert(Array{Float16},mc);
	pMisHps = pMisTemp

	# Rc = Dc-Dobs, where Dc is the clean (wrt Z1,Z2) simulated data
	mergedRc = Array{Array{ComplexF64}}(undef,nfreq);
	for f = 1:nfreq
		mergedRc[f] = reducedDobs[f] .- fetch(Dc[f]);
	end

	println("Misfit with Zero Z2 (our computation): ", misfitCalc(mc,zeros(ComplexF64, (N_nodes, p)),zeros(ComplexF64, (p, size(TEmat,2))),mergedWd ./ sqrt(simSrcDim),mergedRc,nfreq,alpha1,alpha2, pMisHps))
	mergedRcReduced = mergedRc;

	pMisTempFetched = map(fetch, pMisTemp)
	t1 = time_ns();
	srcNum = simSrcDim == 1 ? nsrc : simSrcDim;
	Z2 = calculateZ2(mc,misfitCalc, p, srcNum, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd ./ sqrt(simSrcDim), mergedRcReduced, pMisHps, pMisTempFetched, Z1, alpha2);
	e1 = time_ns();

	print("After First Z2 update: ");
	mis = misfitCalc(mc,Z1,Z2,mergedWd ./ sqrt(simSrcDim),mergedRcReduced,nfreq,alpha1,alpha2, pMisHps);
	obj = objectiveCalc(Z1,Z2,mis,alpha1,alpha2);
	initialMis = mis
	initialObj = obj

	println("mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")

	innerIters = j == 1 ? 5 : 1; # do five in first outer iteration only for better performance
	for iters = 1:innerIters
		###################################################
		#### COMPUTING Z1:
		###################################################
		t1 = time_ns();
		Z1 = calculateZ1(mc,misfitCalc, nfreq, mergedWd ./ sqrt(simSrcDim), mergedRcReduced, pMisHps, Z1, Z2, alpha1, stepReg);
		e1 = time_ns();
		mis = misfitCalc(mc,Z1,Z2,mergedWd ./ sqrt(simSrcDim) ,mergedRcReduced,nfreq,alpha1,alpha2, pMisHps);
		obj = objectiveCalc(Z1,Z2,mis,alpha1,alpha2);
		println("After Z1: mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")

		###################################################
		#### COMPUTING Z2:
		###################################################
		t1 = time_ns();
		Z2 = calculateZ2(mc,misfitCalc, p, srcNum, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd ./ sqrt(simSrcDim), mergedRcReduced, pMisHps, pMisTempFetched, Z1, alpha2);
		e1 = time_ns();
		mis = misfitCalc(mc,Z1,Z2,mergedWd ./ sqrt(simSrcDim) ,mergedRcReduced,nfreq,alpha1,alpha2, pMisHps);
		obj = objectiveCalc(Z1,Z2,mis,alpha1,alpha2);
		println("After Z2: mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")

		# allow extended source to reduce misfit by up to 20%
		if mis < 0.8 * initialMis
			break
		end
	end

	if mis / F_zero > 0.5
			alpha1 = alpha1 / 1.5;
			alpha2 = alpha2 / 1.5;
			println("Ratio mis/F_zero is: ",mis/F_zero,", hence decreasing alphas by 1.5: ",alpha1,",",alpha2);
	end

	if mis / F_zero < 0.3
			alpha1 = alpha1*1.5;
			alpha2 = alpha2*1.5;
			println("Ratio mis/F_zero is: ",mis/F_zero,", hence increasing alpha1 by 1.5: ",alpha1,",",alpha2);
	end


	newSources = originalSources * TEmat + Z1 * Z2;

	pMisTemp = setSourcesSame(pMisTemp,newSources);
	pMisTemp = setDobs(pMisTemp,reducedDobs)
	pMisTemp = setWd(pMisTemp,reducedWd)

	if resultsFilename == ""
		filename = "";
		hisMatFileName = "";
	else
		Temp = splitext(resultsFilename);
		filename = string(Temp[1],"_Cyc",cycle,"_FC",freqIdx,"_",j,"_GN",Temp[2]);
		hisMatFileName  =  string(Temp[1],"_Cyc",cycle,"_FC",freqIdx);
	end

	# Here we set a dump function for GN for this iteracion of FC
	function dumpGN(mc,Dc,iter,pInv,PF)
		dumpFun(mc,Dc,iter,pInv,PF,filename);
	end

	pMisTE = pMisTemp;

	flush(Base.stdout)
	t1 = time_ns();
	if method == "projGN"
		mc,Dc,flag,His = projGNCG(mc,pInv,pMisTE,dumpResults = dumpGN);
	elseif method == "barrierGN"
		mc,Dc,flag,His = barrierGNCG(mc,pInv,pMisTE,rho=1.0,dumpResults = dumpGN);
	end

	e1 = time_ns();
	print("runtime of GN:"); println((e1 - t1)/1.0e9);

	FafterGN = His.F[end];
	println("Computed Misfit with new sources after GN : ",FafterGN);


	if hisMatFileName != ""
		file = matopen(string(hisMatFileName,"_HisGN.mat"), "w");
		### PUT THE CONTENT OF "HIS" INTO THE MAT FILE ###
		His.Dc = [];
		write(file,"His",His);
		close(file);
	end
	His.Dc = []
	push!(HIS,His)
end

pMisTemp = setSourcesSame(pMisTemp,originalSources);
pMisTemp = setDobs(pMisTemp,mergedDobs)
pMisTemp = setWd(pMisTemp,mergedWd)

return mc,Z1,Z2,alpha1,alpha2*simSrcDim,Dc,flag,HIS
end

"""
	function freqCont

	Frequency continuation procedure for running FWI.
	This function runs GaussNewton on misfit functions defined by the frequencies.
"""
function freqCont(freqContParam::FreqContParam)
mc = freqContParam.mc
itersNum = freqContParam.itersNum
originalSources = freqContParam.originalSources
nrcv = freqContParam.nrcv
pInv = freqContParam.pInv
pMis = freqContParam.pMis
windowSize = freqContParam.windowSize
resultsFilename = freqContParam.resultsFilename
dumpFun = freqContParam.dumpFun
Iact = freqContParam.Iact
mback = freqContParam.mback
Z1 = freqContParam.Z1
simSrcDim = freqContParam.simSrcDim
alpha1 = freqContParam.alpha1
alpha2Orig = freqContParam.alpha2Orig
stepReg = freqContParam.stepReg
mode = freqContParam.mode
startFrom = freqContParam.startFrom
endAt = freqContParam.endAt
cycle = freqContParam.cycle
method = freqContParam.method
FWImethod = freqContParam.FWImethod
updateMref = freqContParam.updateMref

Z2 = 0;
Dc = 0;
flag = -1;
HIS = [];
alpha2 = alpha2Orig / simSrcDim;

println("~~~~~~~~~~~~~~~ SEG1  FreqCont: Regs are: ",alpha1,",",alpha2,",",stepReg);
regfun = pInv.regularizer
originalItersNum = itersNum
for freqIdx = startFrom:endAt
	if updateMref
		pInv.mref = copy(mc[:]);
	end
	if mode=="1stInit"
		reqIdx1 = freqIdx;
		if freqIdx > 1
			reqIdx1 = max(2,freqIdx-windowSize+1);
		end
		reqIdx2 = freqIdx;
	else
		reqIdx1 = freqIdx;
		if freqIdx > 1
			reqIdx1 = max(1,freqIdx-windowSize+1);
		end
		reqIdx2 = freqIdx;
	end

	currentProblems = reqIdx1:reqIdx2;
	pMisTemp = pMis[currentProblems];
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	if FWImethod == "FWI"
		pInv.maxIter = itersNum;
		mc,Dc,flag,HIS = standardGNrun(method, mc, simSrcDim, originalSources,
		 		pInv, pMisTemp, HIS, resultsFilename, cycle, freqIdx, dumpFun)
	elseif FWImethod == "FWI_ES"
		pInv.maxIter = 1;
		mc,Z1,Z2,alpha1,alpha2,Dc,flag,HIS = ExtendedSourcesGNRun(
				method, mc, simSrcDim, nrcv, itersNum, originalSources,
				Z1,alpha1, alpha2, stepReg, pInv, pMisTemp, HIS,
				resultsFilename, cycle, freqIdx, dumpFun)
	end

end

pInv.regularizer = regfun;
return mc,Z1,Z2,alpha1,alpha2*simSrcDim,Dc,flag,HIS;
end
