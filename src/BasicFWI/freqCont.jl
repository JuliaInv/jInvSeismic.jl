export freqContBasic;
export freqContTraceEstimation;
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

function freqContZs(mc, pInv::InverseParam, pMis::Array{RemoteChannel},nfreq::Int64, windowSize::Int64,
	Iact,mback::Union{Vector,AbstractFloat,AbstractModel},
			dumpFun::Function, resultsFilename::String, startFrom::Int64 = 1)
Dc = 0;
flag = -1;
HIS = [];
Z = copy(fetch(pMis[1]).pFor.originalSources[:]);
nrec = size(fetch(pMis[1]).pFor.Receivers, 2);
sizeH = size(fetch(pMis[1]).pFor.Ainv[1]);
nsrc = size(Z, 1);
println("NSRC");
println(nsrc);
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
	runningProcs = map(x->x.where, pMis[currentProblems]);
	numOfCurrentProblems = size(currentProblems, 1);
	pForCurrent = Array{RemoteChannel}(undef, numOfCurrentProblems);
	# Zs = Array{}
	t111 = time_ns();
	for s=1:nsrc
		A = sparse([],[],[],sizeH[1], sizeH[2]);
		B = zeros(size(Z[:,s]));
		for i=1:numOfCurrentProblems
			pMisCur = fetch(pMisTemp[i]);
			P = pMisCur.pFor.Receivers;
			beta = 5;
			WdSqr = 2 .*diagm(0 => vec(pMisCur.Wd[:,s])).*diagm(0 => vec(pMisCur.Wd[:,s]));
			# WdSqr = 2 .*vec(pMisCur.Wd[:,s]).*vec(pMisCur.Wd[:,s]);
			println("WDSQR");
			println(size(WdSqr));
			LUcur = pMisCur.pFor.Ainv[i + freqIdx - 1];
			LUcur = LUcur';
			println("LU CUR");
			println(typeof(LUcur));
			println(size(LUcur));
			println("P ");
			println(typeof(P));
			println(size(P));
			# HinvP  = sparse([],[],ComplexF64[],size(P,1), size(P,2));
			HinvP = zeros(ComplexF64, size(P))
			for r = 1: nrec
				println(r);
				HinvP[:, r] = LUcur \ Vector(P[:,r]);
			end
			A = A + HinvP * WdSqr * HinvP' + 2 * beta .* I;
			B = B + 2 * beta .* pMisCur.pFor.originalSources + WdSqr .* HinvP * pMisCur.dobs;
			# Zs =
			# pMisCur.pFor.Sources = Zs;
		end
		# Z[:, s] = A\B;
		# pForTemp = pForpCurrent[i];
		# pForTemp.Sources = Zs;
		println("A/B");
		PMisCur.pFor.Sources[:, s] = A\B;
		println("AFTER");
		pMisTemp[i] = initRemoteChannel(x->x, runningProcs[i], pMisCur);
	end
	s111 = time_ns();
	println("FREQCONT ZS");
	println((s111 - t111)/1.0e9);
	# pMisTemp = getMisfitParam(pForCurrent, WdNew, pMis., SSDFun, Iact, mback);
	# Zs =
	mc,Dc,flag,His = projGN(mc,pInv,pMisTemp,dumpResults = dumpGN);

	clear!(pMisTemp);
	tend = time_ns();
    println("Runtime of freqCont iteration: ");
    println((tend - tstart)/1.0e9);
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
