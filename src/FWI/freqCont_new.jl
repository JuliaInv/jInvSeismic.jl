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
	N_nodes = size(Z1, 1);

	lhs = zeros(ComplexF64, (p,p));

	for i = 1:nwork:numOfCurrentProblems
		mergedSources = zeros(ComplexF64, (N_nodes, nsrc))
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

function misfitCalc2(Z1,Z2,mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2)
	sum = 0.0;
	for i = 1:nwork:numOfCurrentProblems
		# mergedSources = zeros(ComplexF64, (N_nodes, nsrc))
		# mergedDobs = zeros(ComplexF64, (nrec, nsrc))
		# mergedWd = zeros(ComplexF64, (nrec, nsrc))
		# for l=0:(nwork-1)
			# mergedSources[:, currentSrcInd[i+l]] = pMisCurrent[i+l].pFor.Sources
			# mergedDobs[:, currentSrcInd[i+l]] = pMisCurrent[i+l].dobs[:,:,1]
			# mergedWd[:, currentSrcInd[i+l]] = pMisCurrent[i+l].Wd[:,:,1]
		# end
		res = HinvPs[i]' * (mergedSources + Z1 * Z2) - mergedDobs
		sum +=  dot((mergedWd) .* res, (mergedWd).*res);
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
	return sum + alpha * R + stepReg * R
end


function freqContExtendedSources(mc, sources::SparseMatrixCSC, sourcesSubInd::Vector, pInv::InverseParam, pMis::Array{RemoteChannel},contDiv::Array{Int64}, windowSize::Int64,
			resultsFilename::String,dumpFun::Function,mode::String="",startFrom::Int64 = 1,cycle::Int64=0,method::String="projGN")
Dc = 0;
flag = -1;
HIS = [];
println("START EX")

pFor = fetch(pMis[1]).pFor

N_nodes = prod(pFor.Mesh.n .+ 1);


nrec = size(pFor.Receivers, 2);
nsrc = size(pFor.Sources, 2);
alpha1 = 1e1;
alpha2 = 1e1;
stepReg = 1e+1
p = 10;
nwork = nworkers()
nsrc = sum(length.(sourcesSubInd[1:nwork]))
for freqIdx = startFrom:(length(contDiv)-1)
	Z1 = rand(ComplexF64,(N_nodes, p)) .+ 0.01;
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
		mergedSources = zeros(N_nodes, nsrc)
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


		#Calculate HinPs
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
			mergedSources = zeros(ComplexF64, (N_nodes, nsrc))
			mergedDobs = zeros(ComplexF64, (nrec, nsrc))
			mergedWd = zeros(ComplexF64, (nrec, nsrc))
			for l=0:(nwork-1)
				mergedSources[:, currentSrcInd[1+l]] = pMisCurrent[1+l].pFor.Sources
				mergedDobs[:, currentSrcInd[1+l]] = pMisCurrent[1+l].dobs[:,:,1]
				mergedWd[:, currentSrcInd[1+l]] = pMisCurrent[1+l].Wd[:,:,1]
			end
			println("Zero Z2: ", misfitCalc2(zeros(ComplexF64, (N_nodes, p)),zeros(ComplexF64, (p, nsrc)),mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2))
			#Print misfit at start
			println("AT START: ", misfitCalc2(Z1,Z2,mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2))

			Z2 = calculateZ2(misfitCalc2, p, nsrc,nrec,nwork, numOfCurrentProblems, Wd, HinvPs,
				pMisCurrent, currentSrcInd, Z1, alpha2);

			println("misfit after Z2 update:: ", misfitCalc2(Z1,Z2,mergedWd,mergedSources,mergedDobs,numOfCurrentProblems,nwork,alpha1,alpha2))

			#Merge data for each frequency
			numOfFreqs = length(1:nwork:numOfCurrentProblems);
			mergedSourcesArr = Vector{Array}(undef,numOfFreqs)
			mergedDobsArr = Vector{Array}(undef, numOfFreqs)
			mergedWdArr = Vector{Array}(undef, numOfFreqs)
			HPinvsReduced = Vector{Array}(undef, numOfFreqs)
			index = 1
			for i = 1:nwork:numOfCurrentProblems
				mergedSources = zeros(ComplexF64, (N_nodes, nsrc))
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

			rhs = zeros(ComplexF64, size(Z1))
			for i = 1:numOfFreqs
				rhs += MultOpT(HPinvsReduced[i], Rc[i], Z2)
			end
			rhs += stepReg * Z1

			Z1 = KrylovMethods.blockBiCGSTB(x-> MultAll(avgWds, HPinvsReduced, x, Z2, alpha1, stepReg), rhs,x=Z1,maxIter=10, out=2)[1];
			println("misfit at Z1:: ", misfitCalc2())
		end
		# Update the pMis with new sources
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

		# Update all sources
		for i= 1:length(pMis)
			temp = take!(pMis[i])
			temp.pFor.Sources = temp.pFor.OriginalSources + newSrc[:, sourcesSubInd[i]]
			put!(pMis[i], temp)
		end

		#update sources for current problems, maybe not needed?
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
