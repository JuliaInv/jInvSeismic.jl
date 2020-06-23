export freqCont, freqContExtendedSources, freqContExtendedSourcesSS;
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


function calculateZ2(misfitCalc::Function, p::Integer, nsrc::Integer, nfreq::Integer,
	nrec::Integer, nwork::Integer,
	numOfCurrentProblems::Integer, mergedWd::Array, mergedRc::Array, HinvPs::Array,
	pMisCurrent::Array{MisfitParam}, currentSrcInd::Array, Z1::Matrix, alpha::Float64)

	#|| Hz - D||^2 -> Z = (H^TH)^{-1}H^TD

	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	rhs = zeros(ComplexF64, (p, nsrc));
	lhs = zeros(ComplexF64, (p,p));
	for i = 1:nfreq
		meanWd = mean(mergedWd[i])
		lhs .+= (real(meanWd).^2) .* Z1' * HinvPs[i] * HinvPs[i]' * Z1;
		rhs .+= (real(meanWd).^2) .* Z1' * HinvPs[i] * (mergedRc[i]);
	end
	lhs += alpha * I;

	return lhs\rhs;
end


function MyCG(A::Function,b::Array,x::Array,numiter)
r = b-A(x);
p = copy(r);
norms = [norm(r)];
for k=1:numiter
	Ap = A(p);
	alpha = real(dot(r,r)/dot(p,Ap));
	x .+= alpha*p;
	beta = real(dot(r,r));
	r .-= alpha*Ap
	nn = norm(b-A(x));
	norms = [norms; nn];
	beta = real(dot(r,r)) / beta;
	#beta = 0.0;
	p = r + beta*p;
end
return x,norms
end

function MyPCG(A::Function,b::Array,x::Array,M::Array,numiter)
r = b-A(x);
z = M.*r;
p = copy(z);
norms = [norm(r)];
for k=1:numiter
	Ap = A(p);
	alpha = real(dot(z,r)/dot(p,Ap));
	x .+= alpha*p;
	beta = real(dot(z,r));
	r .-= alpha*Ap
	z = M.*r
	nn = norm(b-A(x));
	norms = [norms; nn];
	beta = real(dot(z,r)) / beta;
	#beta = 0.0;
	p = z + beta*p;
end
return x,norms
end




function misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2,HinvPs)
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	sum = 0.0;
	for i = 1:nfreq
		res, = SSDFun((HinvPs[i]' * Z1) * Z2,mergedRc[i],mean(mergedWd[i]) .* ones(size(mergedRc[i])));
		sum += res;
	end
	return sum;
end


function objectiveCalc2(Z1,Z2,misfit,alpha1,alpha2)
	# return misfit + 0.5*alpha1 * norm(Z1)^2 + 0.5*alpha2 * norm(Z2)^2;
	obj = misfit + alpha1 * norm(Z1, 1) + 0.5*alpha2 * norm(Z2)^2;
	return obj
end



function MultOp(HPinv, R, Z2)
	return (HPinv' * R) * Z2
end

function MultOpT(HPinv, R, Z2)
	return HPinv * (R * Z2')
end

function MultAll(avgWds, HPinvs, R, Z1, Z2, alpha, stepReg)
	sum = zeros(ComplexF64, size(R))

	# partials = Array{Array{ComplexF64}}(undef,length(avgWds))
	# function calculateForFreq(HPinv, avgWd, R, Z2)
	# 	return MultOpT(HPinv, (avgWd^2) .* MultOp(HPinv, R, Z2), Z2)
	# end
	# @sync begin
	# 	@async begin
	# 		for k=1:length(avgWds)
	# 			partials[k] = remotecall_fetch(calculateForFreq,
	# 				k % nworkers() + 1,HPinvs[k], avgWds[k], R, Z2);
	# 		end
	# 	end
	# end
	#
	# overallSum = sum(partials)

	for i = 1:length(avgWds)
		sum += MultOpT(HPinvs[i], (avgWds[i]^2) .* MultOp(HPinvs[i], R, Z2), Z2)
	end
	eps = 1e-2
	return sum + ((0.5*alpha)./(abs.(Z1) .+ eps * maximum(abs.(Z1)))).*R + stepReg*R;

	# return sum + (alpha + stepReg)*R;
end

function calculateZ1(misfitCalc::Function, nfreq::Integer, mergedWd::Array, mergedRc::Array, HinvPs::Array, Z1::Matrix,Z2::Matrix, alpha1::Float64,stepReg::Float64)
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	rhs = zeros(ComplexF64, size(Z1));
	for i = 1:nfreq
		rhs .+= (real(mean(mergedWd[i])).^2) .*  MultOpT(HinvPs[i], mergedRc[i], Z2);
	end
	rhs .+= (stepReg) .* Z1

	OP = x-> MultAll(mean.(real.(mergedWd)), HinvPs, x, Z1, Z2, alpha1, stepReg);
	eps = 1e-2
	M = (abs.(Z1) .+ eps * maximum(abs.(Z1)));
	t1=time_ns()
	Z1, = MyPCG(OP,rhs,Z1,M,5);
	e1=time_ns()
	#println("time CG Z1: ", (e1-t1)/1.0e9)
	return Z1;
end

function getMergedData(pMisTemp::Array{RemoteChannel}, nfreq, nsrc, nrcv, nwork, currentSrcInd)
	currentWd = getWd(pMisTemp);
	currentDobs = getDobs(pMisTemp);

	mergedDobs = Array{Array{ComplexF64}}(undef,nfreq);
	mergedWd = Array{Array{ComplexF64}}(undef,nfreq);
	# mergedWd = zeros(ComplexF64,nfreq)
	for f = 1:nfreq
		mergedDobs[f] = zeros(nrcv,nsrc);
		mergedWd[f] = zeros(nrcv,nsrc);

		for l = 1:nwork
			mergedDobs[f][:, currentSrcInd[l]] .= currentDobs[(f-1)*nwork+l]
			mergedWd[f][:, currentSrcInd[l]] = currentWd[(f-1)*nwork+l]
			# mergedWd[f] = mean(currentWd[f*(nwork-1)+l]);
			# println(mergedWd[f],",",currentWd[f*(nwork-1)+l][20,1])
			# println(size(currentWd[f*(nwork-1)+l]))
		end
	end
	return mergedDobs, mergedWd
end

function freqContExtendedSourcesSS(mc,Z1,Z2,newDim,itersNum::Int64,originalSources::SparseMatrixCSC,nrcv, sourcesSubInd::Vector, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,Iact,mback,alpha1,alpha2Orig,
			mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];

alpha2 = alpha2Orig / newDim;

N_nodes = prod(pInv.MInv.n .+ 1);
nsrc = size(originalSources, 2);


stepReg = 0.0;
println("~~~~~~~~~~~~~~~  FreqCont: Regs are: ",alpha1,",",alpha2,",",stepReg);

p = size(Z1,2);
nwork = nworkers()
regfun = pInv.regularizer

# return mc,Z1,Z2,alpha1,alpha2*newDim,Dc,flag,HIS;



for freqIdx = startFrom:(length(contDiv)-1)
	if freqIdx == (length(contDiv)-1) && cycle == 2
		pInv.mref = copy(mc[:]);
		newReg(m,mref,M) 	= wTVReg(m,mref,M,Iact=Iact,C=[]);
		pInv.regularizer = newReg;
	elseif cycle == 2
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
	currentProblems = contDiv[reqIdx1]:contDiv[reqIdx2+1]-1;
	pMisTemp = pMis[currentProblems];
	numOfCurrentProblems = length(pMisTemp);
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");

	currentSrcInd = sourcesSubInd[currentProblems]

	nfreq = div(length(pMisTemp),nwork);

	mergedDobs, mergedWd = getMergedData(pMisTemp, nfreq, nsrc, nrcv, nwork, currentSrcInd)
	OrininalSourcesDivided = Array{SparseMatrixCSC}(undef,length(currentSrcInd));
	for k=1:length(currentSrcInd)
		OrininalSourcesDivided[k] = originalSources[:,currentSrcInd[k]];
	end
	FafterGN = 0.0;
	
	for j = 1:itersNum
		println("============================== New ALM Iter ======================================");
		flush(Base.stdout)

		pMisTemp = setSources(pMisTemp,OrininalSourcesDivided);
		
		# Here we get the current data with clean sources. Also define Ainv (which should be defined already but never mind...).
		t1 = time_ns();
		Dc,F_zero, = computeMisfit(mc,pMisTemp,false);
		e1 = time_ns();
		# print("runtime of computeMisfit orig: "); println((e1 - t1)/1.0e9);


		println("Computed Misfit with orig sources : ",F_zero, " [Time: ",(e1 - t1)/1.0e9," sec]");
		
		if j>1
			if FafterGN < F_zero*0.25
				alpha2 = alpha2*1.5;
				alpha1 = alpha1*1.5;
				println("Ratio FafterGN/F_zero is: ",FafterGN/F_zero,", hence increasing alphas by 1.5: ",alpha1,",",alpha2);
			elseif FafterGN > F_zero*0.75 
				alpha2 = alpha2/1.5;
				alpha1 = alpha1/1.5;
				println("Ratio FafterGN/F_zero is: ",FafterGN/F_zero,", hence decreasing alphas by 1.5: ",alpha1,",",alpha2);
			end
		end
		
		t1 = time_ns();
		HinvPs = computeHinvTRec(pMisTemp[1:nwork:numOfCurrentProblems]);
		e1 = time_ns();
		print("runtime of HINVPs: "); println((e1 - t1)/1.0e9);

		TEmat = rand([-1,1],(nsrc,newDim));

		mergedRc = Array{Array{ComplexF64}}(undef,nfreq); # Rc = Dc-Dobs, where Dc is the clean (wrt Z1,Z2) simulated data
		for f = 1:nfreq
			mergedRc[f] = zeros(ComplexF64,nrcv,nsrc);
			for l = 1:nwork
				mergedRc[f][:, currentSrcInd[l]] .-= fetch(Dc[(f-1)*nwork+l])
			end
			mergedRc[f] .+= mergedDobs[f];
		end
		
		# iterations of alternating minimization between Z1 and Z2
		println("Misfit with Zero Z2 (our computation): ", misfitCalc2(zeros(ComplexF64, (N_nodes, p)),zeros(ComplexF64, (p, nsrc)),mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs))
		mergedRcReduced = map(x-> x*TEmat, mergedRc)
		
		pMisTempFetched = map(fetch, pMisTemp)
		prevObj = 0
		
		doFive = norm(Z2)==0.0;
		
		t1 = time_ns();
		Z2 = calculateZ2(misfitCalc2, p, newDim, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd ./ sqrt(newDim), mergedRcReduced, HinvPs, pMisTempFetched, currentSrcInd, Z1, alpha2);
		e1 = time_ns();
		# print("runtime of calculateZ2: "); println((e1 - t1)/1.0e9);

		print("After First Z2 update: ");
		mis = misfitCalc2(Z1,Z2,mergedWd ./ sqrt(newDim),mergedRcReduced,nfreq,alpha1,alpha2, HinvPs);
		obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
		initialMis = mis
		initialObj = obj
		println("mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")
		for iters = 1:(doFive ? 5 : 1)		

			# if norm(Z1)^2 < 1e-100
				# Z1 = zeros(ComplexF64, size(Z1))
				# Z2 = zeros(ComplexF64, size(Z2))
				# break
			# end
			
			###################################################
			#### COMPUTING Z1:
			###################################################
			t1 = time_ns();
			Z1 = calculateZ1(misfitCalc2, nfreq, mergedWd ./ sqrt(newDim), mergedRcReduced, HinvPs, Z1, Z2, alpha1, stepReg);
			e1 = time_ns();
			# print("runtime of calculateZ1: "); println((e1 - t1)/1.0e9);
			mis = misfitCalc2(Z1,Z2,mergedWd ./ sqrt(newDim) ,mergedRcReduced,nfreq,alpha1,alpha2, HinvPs);
			obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
			println("After Z1: mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")
			
			# roll new TE matrix
			# TEmat = rand([-1,1],(nsrc,newDim));
			# mergedRcReduced = map(x -> x * TEmat, mergedRc);
			
			###################################################
			#### COMPUTING Z2:
			###################################################
			t1 = time_ns();
			Z2 = calculateZ2(misfitCalc2, p, newDim, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd ./ sqrt(newDim), mergedRcReduced, HinvPs, pMisTempFetched, currentSrcInd, Z1, alpha2);
			e1 = time_ns();
			mis = misfitCalc2(Z1,Z2,mergedWd ./ sqrt(newDim) ,mergedRcReduced,nfreq,alpha1,alpha2, HinvPs);
			obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
			println("After Z2: mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2, ", [Time: ",(e1 - t1)/1.0e9," sec]")

			# if abs(obj - prevObj) < 1e-3
				# break
			# end

			# prevObj = obj
			# if mis < 0.5 * initialMis
				# break
			# end
		end

		# get number of non zero elements
		Z1a = abs.(Z1)
		maxZ1 = maximum(Z1a)
		nzcm = length(Z1a[Z1a .> 1e-2*maxZ1])
		nzc = length(Z1a[Z1a .> 1e-3])

		println("Cyc ", cycle, " freqCont ", freqIdx, " iter ", j, ", %nzc ", nzc/prod(size(Z1)), ", %nzcm ", nzcm/prod(size(Z1)))

		

		# write Z1
		Z1abs = zeros(size(Z1,1), 1)
		for i = 1:size(Z1,1)
			Z1abs[i] = norm(Z1[i,:])

		end
		writedlm(string("zs_FC",freqIdx, "_cyc", cycle,"_",j,".mat"),convert(Array{Float16},Z1abs));

		# get Dobs*TEmat
		dobsTE = map(x-> x * TEmat, mergedDobs);

		# get new Wd - matrix filled with mean of old Wd
		sizeWD = size(dobsTE[1])
		wdTE = map(x-> mean(x)/sqrt(newDim) * ones(sizeWD), mergedWd);


		# new sources split indexes
		newSrcInd = [Int[] for i=1:length(currentSrcInd)]
		for k=1:size(TEmat,2)
			for i=1:nfreq
                		append!(newSrcInd[(k % nwork) + 1 + (i-1)*nwork], k)
			end
        end

		#divide Sources, new Dobs (Dobs*TEmat), original Dobs, New Wd, and original Wd
        NewSourcesDivided = Array{SparseMatrixCSC}(undef,length(newSrcInd));
		NewDobsDivided = Array{Array{ComplexF64}}(undef,nwork * length(mergedDobs));
		origDobsDivided = Array{Array{ComplexF64}}(undef,nwork * length(mergedDobs));
		NewWdDivided = Array{Array{ComplexF64}}(undef,nwork * length(mergedWd));
		origWdDivided = Array{Array{ComplexF64}}(undef,nwork * length(mergedWd));

		for k=1:length(NewDobsDivided)
				origDobsDivided[k] = mergedDobs[trunc(Integer, (k-1) / nwork) + 1][:,currentSrcInd[k]];
				origWdDivided[k] = mergedWd[trunc(Integer, (k-1) / nwork) + 1][:,currentSrcInd[k]];
		end

    	for k=1:length(newSrcInd)
			NewDobsDivided[k] = dobsTE[trunc(Integer, (k-1) / nwork) + 1][:,newSrcInd[k]];
			NewWdDivided[k] = wdTE[trunc(Integer, (k-1) / nwork) + 1][:,newSrcInd[k]];
			#NewWdDivided[k] = 1.0./(abs.(NewDobsDivided[k]) .+ 1e-1*mean(abs.(NewDobsDivided[k])));
		end

		newSources = originalSources * TEmat + Z1 * Z2;
    	for k=1:length(newSrcInd)
            	NewSourcesDivided[k] = newSources[:,newSrcInd[k]];
    	end

		pMisTemp = setSources(pMisTemp,NewSourcesDivided);
		pMisTemp = setDobs(pMisTemp,NewDobsDivided)
		pMisTemp = setWd(pMisTemp,NewWdDivided)

		# Dc,F, = computeMisfit(mc,pMisTemp,false);
		# println("Computed Misfit with new sources : ",F);

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
		# pInv.mref = mc[:];

		flush(Base.stdout)
		t1 = time_ns();

		if method == "projGN"
			mc,Dc,flag,His = projGNCG(mc,pInv,pMisTE,dumpResults = dumpGN);
		elseif method == "barrierGN"
			mc,Dc,flag,His = barrierGNCG(mc,pInv,pMisTE,rho=1.0,dumpResults = dumpGN);
		end
		e1 = time_ns();
		print("runtime of GN:"); println((e1 - t1)/1.0e9);

		mc = map(x -> x > 4.2 ? 4.5 : x, mc)
		
		
		FafterGN = His.F[end];
		# println("From his: ",FafterGN) 
		# Dc,FafterGN, = computeMisfit(mc,pMisTE,false);
		println("Computed Misfit with new sources after GN : ",FafterGN);

		# misfitReductionRatio = (FafterGN / F);
		# ERAN: no need to compute the print this. This is shown as part of GN.
		# println("GN misfit reduction ratio : ",misfitReductionRatio);
		 
		# if norm(Z1)^2 > 500
			# alpha1 *= 1.15; #misfitReductionRatio

			# println("icreasing alpha1: ", alpha1)
		# end

		# if alpha1 < 1e-3 && (abs(obj - initialObj) < 10 || (norm(Z1)^2 < 100 && cycle != 2))
			# alpha1 /= 2
		# end

		pMisTemp = setSources(pMisTemp,OrininalSourcesDivided);
		pMisTemp = setDobs(pMisTemp,origDobsDivided)
		pMisTemp = setWd(pMisTemp,origWdDivided)

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

pInv.regularizer = regfun;
return mc,Z1,Z2,alpha1,alpha2*newDim,Dc,flag,HIS;
end

function freqContExtendedSources(mc,Z1,Z2,itersNum::Int64,originalSources::SparseMatrixCSC,nrcv, sourcesSubInd::Vector, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,Iact,mback,alpha1,alpha2,mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];
println("START EX")


N_nodes = prod(pInv.MInv.n .+ 1);
nsrc = size(originalSources, 2);

nwork = nworkers()
println("nworkers: ", nwork)
stepReg = 1e2;
println("FreqCont: Regs are: ",alpha1,",",stepReg);

p = size(Z1,2);
nwork = nworkers()
regfun = pInv.regularizer
for freqIdx = startFrom:(length(contDiv)-1)
	if freqIdx == (length(contDiv)-1) && cycle == 2
		pInv.mref = copy(mc[:]);
		newReg(m,mref,M) 	= wTVReg(m,mref,M,Iact=Iact,C=[]);
		pInv.regularizer = newReg;
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
	currentProblems = contDiv[reqIdx1]:contDiv[reqIdx2+1]-1;
	pMisTemp = pMis[currentProblems];
	numOfCurrentProblems = length(pMisTemp);
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");

	currentSrcInd = sourcesSubInd[currentProblems]
	# pInv.mref = mc[:];
	# iterations of minize Z1,Z2 and then one gauss newton step
	nfreq = div(length(pMisTemp),nwork);

	mergedDobs, mergedWd = getMergedData(pMisTemp, nfreq, nsrc, nrcv, nwork, currentSrcInd)

	meanWd = map(x-> mean(x), mergedWd);
	mergedWd = map(x-> mean(x) .* ones(size(x)), mergedWd)

	for i=1:length(pMisTemp)

		pMisC  = take!(pMisTemp[i])
		wdTmp = pMisC.Wd
		pMisC.Wd = meanWd[(i-1) % nwork + 1] * ones(size(wdTmp));
		put!(pMisTemp[i],pMisC)
	end

	for k=1:length(pMisTemp)
		mergedWdDivided[k] = mergedWd[:,currentSrcInd[k]];
	end
	pMisTemp = setWd(pMisTemp, mergedWd)
	OrininalSourcesDivided = Array{SparseMatrixCSC}(undef,length(currentSrcInd));
	for k=1:length(currentSrcInd)
		OrininalSourcesDivided[k] = originalSources[:,currentSrcInd[k]];
	end

	for j = 1:itersNum
		flush(Base.stdout)

		pMisTemp = setSources(pMisTemp,OrininalSourcesDivided);
		# Here we get the current data with clean sources. Also define Ainv (which should be defined already but never mind...).
		t1 = time_ns();
		Dc,F, = computeMisfit(mc,pMisTemp,false);
		e1 = time_ns();
		print("runtime of computeMisfit orig: "); println((e1 - t1)/1.0e9);
		
		
		println("Computed Misfit with orig sources : ",F);
		t1 = time_ns();
		HinvPs = computeHinvTRec(pMisTemp[1:nwork:numOfCurrentProblems]);
		e1 = time_ns();
		print("runtime of HINVPs: "); println((e1 - t1)/1.0e9);


		mergedRc = Array{Array{ComplexF64}}(undef,nfreq); # Rc = Dc-Dobs, where Dc is the clean (wrt Z1,Z2) simulated data
		for f = 1:nfreq
			mergedRc[f] = zeros(ComplexF64,nrcv,nsrc);
			for l = 1:nwork
				mergedRc[f][:, currentSrcInd[l]] .-= fetch(Dc[(f-1)*nwork+l])
			end
			mergedRc[f] .+= mergedDobs[f];
		end
		# iterations of alternating minimization between Z1 and Z2
		println("Misfit with Zero Z2: ", misfitCalc2(zeros(ComplexF64, (N_nodes, p)),zeros(ComplexF64, (p, nsrc)),mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs))
		mis = misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs);
		obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
		initialMis = mis
		println("At Start: mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2)
		pMisTempFetched = map(fetch, pMisTemp)
		prevObj = 0

		Z2 = calculateZ2(misfitCalc2, p, nsrc, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd, mergedRc, HinvPs, pMisTempFetched, currentSrcInd, Z1, alpha2);

		print("After Z2:");
		mis = misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs);
		obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
		println("mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2)

		for iters = 1:5
			break;
			#Print misfit at start
			println("============================== New Z1-Z2 Iter ======================================");

			Z1 = calculateZ1(misfitCalc2, nfreq, mergedWd, mergedRc, HinvPs, Z1, Z2, alpha1, stepReg);

			print("After Z1:");
			mis = misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs);
			obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
			println("mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2)

			Z2 = calculateZ2(misfitCalc2, p, nsrc, nfreq, nrcv,nwork, numOfCurrentProblems, mergedWd, mergedRc, HinvPs, pMisTempFetched, currentSrcInd, Z1, alpha2);

			print("After Z2:");
			mis = misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2, HinvPs);
			obj = objectiveCalc2(Z1,Z2,mis,alpha1,alpha2);
			println("mis: ",mis,", obj: ",obj,", norm Z2 = ", norm(Z2)^2," norm Z1: ", norm(Z1)^2)

			if abs(obj - prevObj) < 1e-3
				break
			end

			prevObj = obj
			if mis < 0.5 * initialMis
				break
			end


		end

		# get number of non zero elements
		Z1a = abs.(Z1)
		maxZ1 = maximum(Z1a)
		nzcm = size(Z1a[Z1a .> 1e-2*maxZ1])
		nzc = size(Z1a[Z1a .> 1e-3])

		println("Cyc ", cycle, " freqCont ", freqIdx, " iter ", j, "nzc ", nzc, "nzcm ", nzcm)

		# write Z1
		Z1abs = zeros(size(Z1,1), 1)
		for i = 1:size(Z1,1)
			Z1abs[i] = norm(Z1[i,:])

		end
		writedlm(string("zs_FC",freqIdx, "_cyc", cycle,"_",j,".mat"),convert(Array{Float16},Z1abs));

		#Update the pMis with new sources
		newSrc = Z1*Z2
		NewSourcesDivided = Array{SparseMatrixCSC}(undef,length(currentSrcInd));
		for k=1:length(currentSrcInd)
			NewSourcesDivided[k] = originalSources[:,currentSrcInd[k]] + newSrc[:,currentSrcInd[k]];
		end

		pMisTemp = setSources(pMisTemp,NewSourcesDivided);

		Dc,F, = computeMisfit(mc,pMisTemp,false);
		println("Computed Misfit with new sources : ",F);

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

		flush(Base.stdout)
		t1 = time_ns();

		if method == "projGN"
			mc,Dc,flag,His = projGNCG(mc,pInv,pMisTemp,dumpResults = dumpGN);
		elseif method == "barrierGN"
			mc,Dc,flag,His = barrierGNCG(mc,pInv,pMisTE,rho=1.0,dumpResults = dumpGN);
		end
		e1 = time_ns();
		print("runtime of GN:"); println((e1 - t1)/1.0e9);

		mc = map(x -> x > 4.2 ? 4.5 : x, mc)

		Dc,FafterGN, = computeMisfit(mc,pMisTemp,false);
		println("Computed Misfit with new sources after GN : ",FafterGN);

		misfitReductionRatio = (FafterGN / F);
		println("GN misfit reduction ratio : ",misfitReductionRatio);

		pMisTemp = setSources(pMisTemp,OrininalSourcesDivided);

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

pInv.regularizer = regfun;
return mc,Z1,Z2,alpha1,alpha2,Dc,flag,HIS;
end

"""
	Function to calculate MistfitParams of new dimensions after trace estimation
"""
function calculateReducedMisfitParams(mc, currentProblems::UnitRange, pMis::Array{RemoteChannel},
			Iact,mback, TEmat, Dc, nfreq, nsrc, nrcv, nwork, currentSrcInd)

	### need to split the new sources to workers
	numOfCurrentProblems = size(pMis, 1);
	println("typeof Dc:", typeof(Dc))
	eta = 20.0;
	# newDim = 20;
	newDim = size(TEmat, 2)
	runningProcs = map(x->x.where, pMis);
	pMisCurrent = map(fetch, pMis);
	pForpCurrent =  map(x->x.pFor, pMisCurrent);

	# DobsCurrent = map(x->x.dobs[:,:], pMisCurrent);
	# WdCurrent = map(x->x.Wd[:,:], pMisCurrent);

	DobsCurrent, WdCurrent = getMergedData(pMis, nfreq, nsrc, nrcv, nwork, currentSrcInd)
	DobsNew = copy(DobsCurrent[:]);
	nsrc = size(DobsNew[1],2);
	# TEmat = rand([-1,1],(nsrc,newDim));
	WdEta = eta.*WdCurrent;
	# DpCurrent, = getData(vec(mc),pForpCurrent);
	# Dc, = computeMisfit(mc,pMis,false);
	WdNew = Array{Array{ComplexF64}}(undef, numOfCurrentProblems);
	for i=1:numOfCurrentProblems
		DobsTemp = fetch(Dc[i]);
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
		pForTemp.Sources = pForTemp.Sources * TEmat;
		pForReduced[i] = initRemoteChannel(x->x, runningProcs[i], pForTemp);
	end
	return getMisfitParam(pForReduced, WdNew, DobsNew, SSDFun, Iact, mback);
end
