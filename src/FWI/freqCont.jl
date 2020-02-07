export freqCont, freqContExtendedSources;
using JLD
using Multigrid.ParallelJuliaSolver

"""
	function freqCont

	Frequency continuation procedure for running FWI.
	This function runs GaussNewton on misfit functions defined by the continuation division array contDiv.

	Input:
		mc    		- current model
		pInv		- Inverse param
		pMis 		- misfit params (remote)
		contDiv		- continuation division. Assumes that pMis is contiunous with respecto to the division.
					  If the tasks (frequencies) are divided in pMis {1,2} {3,4} {5,6} then contDiv = [1,3,5,7] (similarly to the ptr array in SparseMatrixCSC)
		windowSize  - How many frequencies to treat at once at the most.
		resultsFilename - a filename for saving the intermediate results according to the GN and continuation (FC) iterations (done in dumpFun())
		dumpFun     - a function for plotting, saving and doing all the things with the intermidiate results.
		mode        - either "1stInit" or anything else.
					  1stInit will use the first group of misfits as an initialization and will not include it together with the next group of misfits.
		startFrom   - a start index for the continuation. Usefull when our run broke in the middle of the iterations.
		cycle       - just for identifying different runs when saving the results.
		method      - "projGN" or "barrierGN"

"""
function freqCont(mc, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];
for freqIdx = startFrom:(length(contDiv)-1)
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
	currentProblems = contDiv[reqIdx1]:contDiv[reqIdx2+1]-1;
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	pMisTemp = pMis[currentProblems];
	pInv.mref = mc[:];


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
end
return mc,Dc,flag,HIS;
end


function calculateZ2(misfitCalc::Function, p::Integer, nsrc::Integer,
	nrec::Integer, nwork::Integer,
	numOfCurrentProblems::Integer, Wd::Array, HinvPs::Array,
	pMisCurrent::Array{MisfitParam}, currentSrcInd::Array, Z1::Matrix, alpha::Float64)

	println("misfit at start:: ", misfitCalc())
	rhs = zeros(ComplexF64, (p, nsrc));
	mSizeVec = size(Z1, 1);

	lhs = zeros(ComplexF64, (p,p));

	for i = 1:nwork:numOfCurrentProblems
		mergedSources = zeros(ComplexF64, (mSizeVec, nsrc))
		mergedDobs = zeros(ComplexF64, (nrec, nsrc))
		mergedWd = zeros(ComplexF64, (nrec, nsrc))
		for l=0:(nwork-1)
			mergedSources[:, currentSrcInd[i+l]] = pMisCurrent[i+l].pFor.Sources
			mergedDobs[:, currentSrcInd[i+l]] = pMisCurrent[i+l].dobs[:,:,1]
			mergedWd[:, currentSrcInd[i+l]] = Wd[i+l]
		end
		pm = pMisCurrent[i]
		lhs += (abs(mean(mergedWd))^2) .* Z1' * HinvPs[i] * HinvPs[i]' * Z1;
		rhs += (abs(mean(mergedWd))^2) .* Z1' * HinvPs[i] * (-HinvPs[i]' * mergedSources + mergedDobs);
	end

	lhs += alpha * I;

	return lhs\rhs;
end



function freqContExtendedSources(mc, sources::SparseMatrixCSC, sourcesSubInd::Vector, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];
println("START EX")

pFor = fetch(pMis[1]).pFor
mSizeMat = pFor.Mesh.n .+ 1;
m = mSizeMat[1];
n = mSizeMat[2];
mSizeVec = mSizeMat[1] * mSizeMat[2];

nrec = size(pFor.Receivers, 2);
nsrc = size(pFor.Sources, 2);
alpha1 = 1e1;
alpha2 = 1e1;
stepReg = 1e+1
p = 10;
nwork = nworkers()
nsrc = sum(length.(sourcesSubInd[1:nwork]))
for freqIdx = startFrom:(length(contDiv)-1)
	Z1 = rand(ComplexF64,(m*n, p)) .+ 0.01;
	Z2 = rand(ComplexF64, (p, nsrc)) .+ 0.01;
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
	currentProblems = contDiv[reqIdx1]:contDiv[reqIdx2+1]-1;
	numOfCurrentProblems = length(currentProblems);
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	pMisTemp = pMis[currentProblems];
	currentSrcInd = sourcesSubInd[currentProblems]
	pInv.mref = mc[:];
	# iterations of minize Z1,Z2 and then one gauss newton step
	for j = 1:5
		mergedSources = zeros(mSizeVec, nsrc)
		mergedDobs = zeros(nrec, nsrc)
		mergedWd = zeros(nrec, nsrc)

		pMisCurrent = map(fetch, pMisTemp);
		for pm in pMisCurrent
			pm.pFor.Sources = pm.pFor.OriginalSources
		end
		Wd = map(pm -> pm.Wd[:,:,1], pMisCurrent)
		HinvPs = Vector{Array}(undef, numOfCurrentProblems);
		t1 = time_ns();
		pForpCurrent =  map(x->x.pFor, pMisCurrent);

		Dp,pForp = getData(vec(mc), pForpCurrent);

		for freqs = 1:nworkers():numOfCurrentProblems
			pForCurrent = pForp[freqs]
			pMisCurrent[freqs].pFor = pForCurrent
			Ainv = pForCurrent.ForwardSolver;
			result, = Multigrid.ParallelJuliaSolver.solveLinearSystem(spzeros(ComplexF64,0,0), complex(Matrix(pForCurrent.Receivers)), Ainv, 1)

			for i=0:(nworkers() -1)
				if (i + freqs)<= numOfCurrentProblems
					HinvPs[freqs + i] = result
				end
			end

			println("HINVP done");
		end

		e1 = time_ns();
		println("runtime of HINVPs");
		println((e1 - t1)/1.0e9);

		# iterations of alternating minimization between Z1 and Z2
		for iters = 1:10
			function misfitCalc2()
				sum = 0;

				for i = 1:nwork:numOfCurrentProblems
					mergedSources = zeros(ComplexF64, (mSizeVec, nsrc))
					mergedDobs = zeros(ComplexF64, (nrec, nsrc))
					mergedWd = zeros(ComplexF64, (nrec, nsrc))
					for l=0:(nwork-1)
						mergedSources[:, currentSrcInd[i+l]] = pMisCurrent[i+l].pFor.Sources
						mergedDobs[:, currentSrcInd[i+l]] = pMisCurrent[i+l].dobs[:,:,1]
						mergedWd[:, currentSrcInd[i+l]] = pMisCurrent[i+l].Wd[:,:,1]
					end

					res = HinvPs[i]' * (mergedSources + Z1 * Z2) - mergedDobs
					sum +=  dot((mergedWd) .* res, (mergedWd).*res);
				end

				sum	+= alpha1 * norm(Z1)^2 + alpha2 * norm(Z2)^2;
				return sum;
			end

		# Print misfit without Z1,Z2
		Z2old = Z2
		Z1old = Z1
		Z2 = zeros(ComplexF64, (p, nsrc));
		Z1 = zeros(ComplexF64, (mSizeVec, p));
		println("Zero Z2: ", misfitCalc2())
		Z2 = Z2old
		Z1 = Z1old


		mergedSources = zeros(ComplexF64, (mSizeVec, nsrc))
		mergedDobs = zeros(ComplexF64, (nrec, nsrc))
		mergedWd = zeros(ComplexF64, (nrec, nsrc))
		for l=0:(nwork-1)
			mergedSources[:, currentSrcInd[1+l]] = pMisCurrent[1+l].pFor.Sources
			mergedDobs[:, currentSrcInd[1+l]] = pMisCurrent[1+l].dobs[:,:,1]
			mergedWd[:, currentSrcInd[1+l]] = pMisCurrent[1+l].Wd[:,:,1]
		end

		# save("ext2.jld", "hps", HinvPs, "dobs", mergedDobs, "wd", mergedWd,
		# "src", mergedSources)
		#Print misfit at start
		println("AT START: ", misfitCalc2())

		Z2 = calculateZ2(misfitCalc2, p, nsrc,nrec,nwork, numOfCurrentProblems, Wd, HinvPs,
			pMisCurrent, currentSrcInd, Z1, alpha2);

		println("misfit at Z2:: ", misfitCalc2())



		function MultOp(HPinv, R, Z2)
			return HPinv' * R * Z2
		end

		function MultOpT(HPinv, R, Z2)
			return HPinv * R * Z2'
		end

		function MultAll(avgWds, HPinvs, R, Z2, alpha, stepReg)
			sum = zeros(ComplexF64, size(R))
			for i = 1:length(avgWds)
				sum += MultOpT(HPinvs[i], (avgWds[i]^2) .* MultOp(HPinvs[i], R, Z2), Z2)
			end
			return sum + alpha * R + stepReg * R
		end

		numOfFreqs = length(1:nwork:numOfCurrentProblems);
		mergedSourcesArr = Vector{Array}(undef,numOfFreqs)
		mergedDobsArr = Vector{Array}(undef, numOfFreqs)
		mergedWdArr = Vector{Array}(undef, numOfFreqs)
		HPinvsReduced = Vector{Array}(undef, numOfFreqs)
		index = 1
		for i = 1:nwork:numOfCurrentProblems
			mergedSources = zeros(ComplexF64, (mSizeVec, nsrc))
			mergedDobs = zeros(ComplexF64, (nrec, nsrc))
			mergedWd = zeros(ComplexF64, (nrec, nsrc))
			for l=0:(nwork-1)
				mergedSources[:, currentSrcInd[i+l]] = pMisCurrent[i+l].pFor.Sources
				mergedDobs[:, currentSrcInd[i+l]] = pMisCurrent[i+l].dobs[:,:,1]
				mergedWd[:, currentSrcInd[i+l]] = Wd[i+l]
			end
			mergedSourcesArr[index] = mergedSources
			mergedDobsArr[index] = mergedDobs
			mergedWdArr[index] = mergedWd
			HPinvsReduced[index] = HinvPs[i]
			index += 1
		end

		avgWds = Vector{Float64}(undef, numOfFreqs)

		Rc = Vector{Array}(undef, numOfFreqs)
		for i=1:numOfFreqs
			avgWds[i] = abs(mean(mergedWdArr[i]))
			Rc[i] = (avgWds[i]^2) .* (mergedDobsArr[i] - HPinvsReduced[i]' * mergedSourcesArr[i])
		end

		# Rsum = sum(Rc)

		rhs = zeros(ComplexF64, size(Z1))
		for i = 1:numOfFreqs
			rhs += MultOpT(HPinvsReduced[i], Rc[i], Z2)
		end
		rhs += stepReg * Z1

		# save("ext.jld", "hps", HPinvsReduced, "dobs", mergedDobsArr, "wd", mergedWdArr,
		# "src", mergedSourcesArr)
		Z1 = KrylovMethods.blockBiCGSTB(x-> MultAll(avgWds, HPinvsReduced, x, Z2, alpha1, stepReg), rhs,x=Z1,maxIter=10, out=2)[1];
		# throw("abs")
		println("misfit at Z1:: ", misfitCalc2())

	end

	newSrc = Z1*Z2
	for i=1:numOfCurrentProblems
		pMisCurrent[i].pFor.Sources = pMisCurrent[i].pFor.OriginalSources + newSrc[:,currentSrcInd[i]]
	end
	pForpCurrent =  map(x->x.pFor, pMisCurrent);
	Dp,pForp = getData(vec(mc), pForpCurrent);

	for freqs = 1:numOfCurrentProblems
		pForCurrent = pForp[freqs]
		pMisCurrent[freqs].pFor = pForCurrent
	end

	for i= 1:length(pMis)
		temp = take!(pMis[i])
		temp.pFor.Sources = temp.pFor.OriginalSources + newSrc[:, sourcesSubInd[i]]
		put!(pMis[i], temp)
	end

	for i=1:numOfCurrentProblems
		temp = take!(pMisTemp[i])
		temp.pFor.Sources = temp.pFor.OriginalSources + newSrc[:, currentSrcInd[i]]
		put!(pMisTemp[i], pMisCurrent[i])
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

end
return mc,Dc,flag,HIS;
end

#
# function freqContExtendedSources(mc, pInv::InverseParam, pMis::Array{RemoteChannel},
# 	nfreq::Int64, windowSize::Int64, Iact,
# 	mback::Union{Vector,AbstractFloat,AbstractModel, Array{Float64,1}},
# 	dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
# Dc = 0;
# flag = -1;
# HIS = [];
#
# pFor = fetch(pMis[1]).pFor
# mSizeMat = pFor.Mesh.n .+ 1;
# m = mSizeMat[1];
# n = mSizeMat[2];
# mSizeVec = mSizeMat[1] * mSizeMat[2];
# # Z = copy(pFor.originalSources);
# # sizeQ = size(Z);
# nrec = size(pFor.Receivers, 2);
# sizeH = size(pFor.Ainv[1]);
# pFor = nothing
# nsrc = size(pFor.Sources, 2);
# alpha = 2e-3;
# p = 10;
#
# for freqIdx = startFrom:nfreq
# 	Z1 = rand(ComplexF64,(m*n, p)) .+ 0.01;
# 	Z2 = rand(ComplexF64, (p, nsrc)) .+ 0.01;
# 	println("start freqCont Zs iteration from: ", freqIdx)
# 	tstart = time_ns();
# 	reqIdx1 = freqIdx;
# 	if freqIdx > 1
# 		reqIdx1 = max(1,freqIdx-windowSize+1);
# 	end
# 	reqIdx2 = freqIdx;
# 	currentProblems = reqIdx1:reqIdx2;
#
#
# 	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
# 	runningProcs = map(x->x.where, pMis[currentProblems]);
#
#
# 	for j=1:5
#
# 		pMisCurrent = map(fetch, pMis[currentProblems]);
# 		pForpCurrent =  map(x->x.pFor, pMisCurrent);
# 		Dp,pForp = getData(vec(mc), pForpCurrent);
#
# 		pForCurrent = map(x->fetch(x), pForp);
# 		numOfCurrentProblems = size(currentProblems, 1);
# 		map((pm,pf) -> pm.pFor = pf , pMisCurrent, pForCurrent);
# 		HinvPs = Vector{Array}(undef, numOfCurrentProblems);
# 		pMisTemp = Array{RemoteChannel}(undef, length(currentProblems));
# 		t1 = time_ns();
# 		for freqs = 1:numOfCurrentProblems
# 			U,Ainv = solveLinearSystem(H,U,Ainv,0)
# 			HinvPs[freqs] = (pForCurrent[freqs].Ainv)' \ Matrix(pForCurrent[freqs].Receivers);
# 			println("HINVP done");
# 		end
# 		e1 = time_ns();
# 		println("runtime of HINVPs");
# 		println((e1 - t1)/1.0e9);
#
# 		if resultsFilename == ""
# 				filename = "";
# 			else
# 				Temp = splitext(resultsFilename);
# 				filename = string(Temp[1],"_FC",freqIdx,"_",j,"_GN",Temp[2]);
# 		end
#
# 		function dumpGN(mc,Dc,iter,pInv,PF)
# 			dumpFun(mc,Dc,iter,pInv,PF,filename);
# 		end
#
# 		t2 = time_ns();
# 		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);
# 		e2 = time_ns();
# 		println("Misfit B4 mzs at GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
# 		println("runtime of compute misfit");
# 		println((e2 - t2)/1.0e9);
#
# 		t3 = time_ns();
#
# 		pMisArr = map(pm -> fetch(pm), pMisCurrent);
# 		Ap = Vector{Matrix}(undef, numOfCurrentProblems);
# 		diags = Vector{SparseMatrixCSC}(undef, numOfCurrentProblems);
# 		B = Vector{Vector}(undef, numOfCurrentProblems);
#
# 		toVec(mat) = reshape(mat, mSizeVec);
# 		toMat(vec) = reshape(vec, tuple(mSizeMat...));
#
# 		Wd = map(pm -> pm.Wd[:,:,1], pMisCurrent)
# 		function misfitCalc2()
# 			sum = 0;
# 			for i = 1:numOfCurrentProblems
# 				sum += (mean(Wd[i])^2) .* norm(HinvPs[i]' * (pMisCurrent[i].pFor.Sources + Z1 * Z2) - pMisCurrent[i].dobs[:,:,1])^2;
# 			end
#
# 			sum	+= alpha * norm(Z1)^2 + alpha * norm(Z2)^2;
# 			return sum;
# 		end
#
# 		Z2 = calculateZ2(misfitCalc2, p, nsrc, numOfCurrentProblems, Wd, HinvPs,
# 			pMisCurrent, Z1, alpha);
#
# 		println("misfit at Z2:: ", misfitCalc2())
#
# 		function multOP(R,HinvP)
# 			return HinvP' * R * Z2;
# 		end
#
# 		function multOPT(R,HinvP)
# 			return HinvP * R * Z2';
# 		end
#
# 		function multAll(x)
# 			sum = zeros(ComplexF64, (mSizeVec, p));
#
# 			for i = 1:numOfCurrentProblems
# 				sum += (mean(Wd[i])^2) .* multOPT(multOP(x, HinvPs[i]), HinvPs[i]);
# 			end
#
# 			sum += alpha * x;
# 			return sum;
# 		end
#
# 		rhs = zeros(ComplexF64, (mSizeVec, p));
# 		for i = 1:numOfCurrentProblems
# 			pm = pMisCurrent[i]
# 			rhs += (mean(Wd[i])^2).*multOPT(-HinvPs[i]' * pm.pFor.Sources + pm.dobs[:,:,1], HinvPs[i]);
# 		end
#
# 		Z1 = KrylovMethods.blockBiCGSTB(x-> real(multAll(x)), real(rhs))[1];
#
# 		for i = 1:numOfCurrentProblems
# 			pMisCurrent[i].pFor.Sources += Z1 * Z2;
# 		end
#
# 		println("misfit at Z1:: ", misfitCalc2())
#
# 		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc, pMisCurrent);
#
# 		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
# 		pMispCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
# 		for i=1:numOfCurrentProblems
# 			pMispCurrent[i] = initRemoteChannel(x->x, runningProcs[i], pMisCurrent[i]);
# 		end
# 		pInv.mref = mc[:];
# 		t4 = time_ns();
# 		mc,Dc,flag,His = projGNCG(mc,pInv,pMispCurrent,dumpResults = dumpGN);
# 		e4 = time_ns();
# 		println((t4 - e4)/1.0e9);
# 		Dc,F,dF,d2F,pMisNone,times,indDebit = computeMisfit(mc,pMisCurrent);
#
# 		println("Misfit after GN ", j, "frequncy idx: ", freqIdx, " Is: ", F);
#
# 		pMis[currentProblems] = pMispCurrent;
# 		clear!(pMispCurrent);
# 	end
#
# 	tend = time_ns();
#     println("Runtime of freqCont iteration: ");
#     println((tend - tstart)/1.0e9);
# 	global inx = inx + 1;
#
# end
# return mc,Dc,flag,HIS;
# end
