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
	
	#|| Hz - D||^2 -> Z = (H^TH)^{-1}H^TD
	
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	rhs = zeros(ComplexF64, (p, nsrc));
	lhs = zeros(ComplexF64, (p,p));
	for i = 1:nfreq
		lhs .+= (real(mergedWd[f]).^2) .* Z1' * HinvPs[i] * HinvPs[i]' * Z1;
		rhs .+= (real(mergedWd[f]).^2) .* Z1' * HinvPs[i] * (-mergedRc[i]);
	end
	lhs += alpha * I;

	return lhs\rhs;
end

function misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,alpha1,alpha2,HinvPs)
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	sum = 0.0;
	for i = 1:nfreq
		res = SSDfun((HinvPs[i]' * Z1) * Z2,mergedRc[i],mergedWd[i]);
		sum += res;
	end
	sum	+= alpha1 * norm(Z1)^2 + alpha2 * norm(Z2)^2;
	return sum;
end

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
	return sum + (alpha+stepReg)*R;
end

function calculateZ1(misfitCalc::Function, nfreq::Integer, mergedWd::Array, HinvPs::Array, Z1::Matrix,Z2::Matrix, alpha1::Float64,stepReg::Float64)
	## HERE WE  NEED TO MAKE SURE THAT Wd is equal in its real and imaginary parts.
	rhs = zeros(ComplexF64, size(Z1));
	for i = 1:nfreq
		rhs .+= (real(mergedWd[f]).^2) .*  MultOpT(HinvPs[i], -mergedRc[i], Z2);
	end
	rhs .+= (stepReg) .* Z1
	Z1 = KrylovMethods.blockBiCGSTB(x-> MultAll(avgWds, HPinvsReduced, x, Z2, alpha1, stepReg), rhs,x=copy(Z1),maxIter=50, out=2)[1];
	return Z1;
end


function freqContExtendedSources(mc,Z1,Z2,originalSources::SparseMatrixCSC,nrcv, sourcesSubInd::Vector, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];
println("START EX")


N_nodes = prod(pInv.MInv.n .+ 1);
nsrc = size(originalSources, 2);

alpha1 = 1e1;
alpha2 = 1e1;
stepReg = 1e+1
p = size(Z1,2);
nwork = nworkers()



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
	pMisTemp = pMis[currentProblems];
	numOfCurrentProblems = length(pMisTemp);
	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
	
	currentSrcInd = sourcesSubInd[currentProblems]
	pInv.mref = mc[:];
	# iterations of minize Z1,Z2 and then one gauss newton step
	
	currentWd = getWd(pMisTemp);
	currentDobs = getDobs(pMisTemp);
	nfreq = div(length(pMisTemp),nwork);
	
	mergedDobs = Array{Array{ComplexF64}}(undef,nfreq);
	#mergedWd = Array{Array{ComplexF64}}(undef,nfreq);
	mergedWd = zeros(ComplexF64,nfreq)
	for f = 1:nfreq
		mergedDobs[f] = zeros(nrcv,nsrc);
		#mergedWd[f] = zeros(nrcv,nsrc);
		for l = 1:nwork
			mergedDobs[f][:, currentSrcInd[l]] .= currentDobs[f*(nwork-1)+l]
			# mergedWd[f][:, currentSrcInd[l]] = currentWd[f*(nwork-1)+l]
			mergedWd[f] .= mean(currentWd[f*(nwork-1)+l]);
		end
	end
	OrininalSourcesDivided = Array{SparseMatrixCSC}(undef,length(currentSrcInd));
	for k=1:length(currentSrcInd)
		OrininalSourcesDivided = originalSources(:,currentSrcInd[k]);
	end
	
	for j = 1:5
		setSources(pMisTemp,OrininalSourcesDivided);
		# Here we get the current data with clean sources. Also define Ainv (which should be defined already but never mind...). 
		t1 = time_ns();
		Dc, = computeMisfit(mc,pMisTemp,false);
		e1 = time_ns();
		println("runtime of misfit");
		println((e1 - t1)/1.0e9);

		t1 = time_ns();
		HinvPs = computeHinvTRec(pMisTemp(1:nwork:numOfCurrentProblems));
		e1 = time_ns();
		println("runtime of HINVPs");
		println((e1 - t1)/1.0e9);
		
		mergedRc = Array{Array{ComplexF64}}(undef,nfreq); # Rc = Dc-Dobs, where Dc is the clean (wrt Z1,Z2) simulated data
		for f = 1:nfreq
			mergedRc[f] = zeros(ComplexF64,nrcv,nsrc);
			for l = 1:nwork
				mergedRc[f][:, currentSrcInd[l]] = fetch(Dc[f*(nwork-1)+l])
			end
			mergedRc[f] .-= mergedDobs[f];
		end
		# iterations of alternating minimization between Z1 and Z2
		println("Zero Z2: ", misfitCalc2(zeros(ComplexF64, (N_nodes, p)),zeros(ComplexF64, (p, nsrc)),mergedWd,mergedRc,nfreq,nwork,alpha1,alpha2))
		for iters = 1:10
			#Print misfit at start
			println("AT START: ", misfitCalc2(Z1,Z2,mergedWd,mergedRc,nfreq,nwork,alpha1,alpha2))

			Z2 = calculateZ2(misfitCalc2, p, nsrc,nrcv,nwork, numOfCurrentProblems, Wd, HinvPs, pMisCurrent, currentSrcInd, Z1, alpha2);

			println("misfit after Z2 update:: ", misfitCalc2(Z1,Z2,mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2))

			Z1 = calculateZ1(misfitCalc2, p, nsrc,nrcv,nwork, numOfCurrentProblems, Wd, HinvPs, pMisCurrent, currentSrcInd, Z1, alpha2);
			
			println("misfit after Z1 update:: ", misfitCalc2(Z1,Z2,mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2))
		end
		
		# Update the pMis with new sources
		newSrc = Z1*Z2
		NewSourcesDivided = Array{SparseMatrixCSC}(undef,length(currentSrcInd));
		for k=1:length(currentSrcInd)
			NewSourcesDivided[k] = originalSources(:,currentSrcInd[k]) + newSrc(:,currentSrcInd[k]);
		end
		setSources(pMisTemp,NewSourcesDivided);

		if resultsFilename == ""
			filename = "";
			hisMatFileName = "";
		else
			Temp = splitext(resultsFilename);
			if cycle==0
				filename = string(Temp[1],"_FC",freqIdx,"_",j,"_GN",Temp[2]);
				hisMatFileName  = string(Temp[1],"_FC",freqIdx);
			else
				filename = string(Temp[1],"_Cyc",cycle,"_FC",freqIdx,"_",j,"_GN",Temp[2]);
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
return mc,Z1,Z2,Dc,flag,HIS;
end
