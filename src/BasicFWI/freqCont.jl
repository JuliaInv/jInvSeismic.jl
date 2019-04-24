export freqContBasic;
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
		startFrom   - a start index for the continuation. Usefull when our run broke in the middle of the iterations.

"""
function freqContBasic(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];
println("FREQCONT CHANGE3")
for freqIdx = startFrom:nfreq
	println("start freqCont iteration from: ", freqIdx)
	tstart = time_ns();


	# pMis = getMisfitParam(pFor, Wd, Dobs, misfun, Iact, mback);
	#
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
pInv.maxIter = 10;
mc,Dc,flag,His = projGNCG(mc,pInv,pMis,dumpResults = dumpGN);
return mc,Dc,flag,HIS;
end

function freqContBasic(mc, pFor, Dobs::Array, Wd::Array, pInv::InverseParam, misfun::Function,
                            Iact,mback::Union{Vector,AbstractFloat,AbstractModel}
							,nfreq::Int64, windowSize::Int64,
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];

println("NEW FREQ CONT");
for freqIdx = startFrom:nfreq
	println("start freqCont iteration from: ", freqIdx)
	tstart = time_ns();


	eta = 20.0;
	DobsNew = copy(Dobs[:])
	println("size DobsNew: ", size(DobsNew) );
	println("size DobsNew: ", size(DobsNew[1], 2) );
	WdEta = eta.*Wd;
	DpNew, = getData(vec(mc), pFor);
	DobsTemp = Array{Array}(undef, nfreq);
	for i=1:nfreq
		DobsTemp[i] = fetch(DpNew[i]);
	end
	println(size(DobsTemp[1][:,1]))
	for i=1:nfreq

		for s=1:size(DobsNew[1], 2)
			etaM = 2 .*diagm(0 => vec(Wd[1][:,1])).*diagm(0 => vec(Wd[1][:,1]));
			A = (2 .*I + etaM)
			# println(size(A))
			#
			# println(size(A\DobsTemp[i][:,s]))
			# println(size(Dobs[i][:,s]))
			# println(size(etaM))
			b = (etaM)*(A\Dobs[i][:,s])
			DobsNew[i][:,s] = 2 .* (A\DobsTemp[i][:,s]) + b;
		end
	end
	WdNew = Array{Array}(undef, nfreq);
	for i=1:nfreq
		WdNew[i] = ones((size(DobsNew[1], 1), size(DobsNew[1], 2)));
	end
	println(size(Wd), " ,", size(WdNew));
	println(size(Wd[1]), " ,", size(WdNew[1]));
	println(size(Dobs), " ,", size(DobsNew));
	println(size(Dobs[1]), " ,", size(DobsNew[1]));
	pMis = getMisfitParam(pFor, WdNew, DobsNew, misfun, Iact, mback);
	#
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
pInv.maxIter = 10;
mc,Dc,flag,His = projGNCG(mc,pInv,pMis,dumpResults = dumpGN);
return mc,Dc,flag,HIS;
end





function freqContBasic(mc, pInv::InverseParam, pMis::MisfitParam,nfreq::Int64, windowSize::Int64,
			dumpFun::Function,startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];

# for freqIdx = startFrom:nfreq
# 	reqIdx1 = freqIdx;
# 	if freqIdx > 1
# 		reqIdx1 = max(1,freqIdx-windowSize+1);
# 	end
# 	reqIdx2 = freqIdx;
# 	currentProblems = reqIdx1:reqIdx2;
# 	println("\n======= New Continuation Stage: selecting continuation batches: ",reqIdx1," to ",reqIdx2,"=======\n");
# 	pMisTemp = pMis[currentProblems];
# 	pInv.mref = mc[:];
#
# 	# Here we set a dump function for GN for this iteracion of FC
# 	function dumpGN(mc,Dc,iter,pInv,PF)
# 		dumpFun(mc,Dc,iter,pInv,PF);
# 	end

	println("NEW STUFF");
	mc,Dc,flag,His = projGNCG(mc,pInv,pMis,dumpResults = dumpGN);

	# clear!(pMisTemp);
# end
return mc,Dc,flag,HIS;
end
