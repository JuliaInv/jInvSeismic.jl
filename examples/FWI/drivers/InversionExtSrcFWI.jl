using Revise
using Distributed
using DelimitedFiles
using MAT
using Multigrid.ParallelJuliaSolver
using jInvSeismic.FWI
using jInvSeismic.Utils
using Helmholtz
using Statistics
using jInv.InverseSolve
using jInv.LinearSolvers
using Multigrid

NumWorkers = 4;
if nworkers() == 1
	addprocs(NumWorkers);
elseif nworkers() < NumWorkers
 	addprocs(NumWorkers - nworkers());
end

@everywhere begin
	using jInv.InverseSolve
	using jInv.LinearSolvers
	using jInvSeismic.FWI
	using jInv.Mesh
	using Multigrid.ParallelJuliaSolver
	using jInv.Utils
	using DelimitedFiles
	using jInv.ForwardShare
	using KrylovMethods
end

plotting = true;
if plotting
	using jInvVisPyPlot
	using PyPlot
	close("all")
end

@everywhere FWIDriversPath = "./";
include(string(FWIDriversPath,"prepareFWIDataFiles.jl"));
include(string(FWIDriversPath,"setupFWI.jl"));

dataDir 	= pwd();
resultsDir 	= pwd();
modelDir 	= pwd();

@enum Model SEG MARMOUSI OVERTHRUST UP
chosenModel = MARMOUSI # choose one of the above models

@enum Method FWI FWI_ES FWI_ES_SS FWI_SS
chosenMethod = FWI_ES_SS

if chosenMethod == FWI || chosenMethod == FWI_ES
	simSrcDim = 1;
else
	simSrcDim = 16;
end

########################################################################################################
windowSize = 4; # frequency continuation window size
maxBatchSize = 256; # use smaller value for 3D
useFilesForFields = false; # wheter to save fields to files

if chosenModel == SEG || chosenModel == UP
	########## uncomment block for SEG ###############
	 dim     = 2;
	 pad     = 30;
	 jumpSrc = 5;
	 jumpRcv = 1;
	 newSize = [600,300];

	if chosenModel == SEG
	 (m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMref(modelDir,
	 	"examples/SEGmodel2Dsalt.dat",dim,pad,[0.0,13.5,0.0,4.2],newSize,1.752,2.9);
	else
	 (m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMref(modelDir,
	 	"examples/SEGmodel2D_up.dat",dim,pad,[0.0,13.5,0.0,4.2],newSize,1.752,2.9, false);
	end
	omega = [3.0,3.3,3.6,3.9,4.2,4.5,5.0,5.5,6.5]*2*pi;
	offset  = newSize[1];
	println("Offset is: ",offset," cells.")
	alpha1 = 1e0;
	alpha2 = 1e2;
	stepReg = 5e1;
	EScycles = 2;
	cgit = 7;

	freqContSweeps = 5;
	freqRanges = [(1,4), (1,4), (4,length(omega)), (4,length(omega)),
			(length(omega), length(omega))];
	regularizations = ["high", "high", "low", "low", "low"];
	GNiters = [20, 20, 15 ,15, 100];

elseif chosenModel == MARMOUSI
	include(string(FWIDriversPath,"generateMrefMarmousi.jl"));
	omega = [3.0,3.5,4.0,4.5,5.0,5.5,6.5,7.5,8.5]*2*pi;
	alpha1 = 1e0;
	alpha2 = 1e2;
	stepReg = 5e1;
	EScycles = 1;
	cgit = 5;

	freqContSweeps = 4;
	freqRanges = [(1,4), (4,length(omega)), (4,length(omega)),
			(length(omega), length(omega))];
	regularizations = ["low", "low", "low", "low"];
	GNiters = [10, 10, 10, 100];

elseif chosenModel == OVERTHRUST
	include(string(FWIDriversPath,"generateMrefOverthrust.jl"));
	omega = [2.5,3.0,3.5,4.0,5.0]*2*pi;

	alpha1 = 5e0;
	alpha2 = 5e2;
	stepReg = 5e1;
	EScycles = 1;
	cgit = 5;

	freqContSweeps = 4;
	freqRanges = [(1,3), (4,length(omega)), (4,length(omega)),
			(length(omega), length(omega))];
	regularizations = ["high", "low", "low", "low"];
	GNiters = [10, 15, 15, 20];
else
	println("Wrong model chosen, driver shutting down");
	exit(1);
end

# ###################################################################################################################
dataFilenamePrefix = string(dataDir,"/DATA_",tuple((Minv.n)...));
resultsFilename = string(resultsDir,"/FWI_",tuple((Minv.n)...));
#######################################################################################################################
writedlm(string(resultsFilename,"_mtrue.dat"),convert(Array{Float16},m));
writedlm(string(resultsFilename,"_mref.dat"),convert(Array{Float16},mref));
resultsFilename = string(resultsFilename,".dat");

println("omega*maximum(h): ",omega*maximum(Minv.h)*sqrt(maximum(1.0./(boundsLow.^2))));
ABLpad = pad + 4;
if chosenModel == OVERTHRUST
	levels      = 3;
	numCores 	= 16;
	BLAS.set_num_threads(numCores);
	maxIter     = 30;
	relativeTol = 1e-6;
	relaxType   = "Jac";
	relaxParam  = 0.75;
	relaxPre 	= 2;
	relaxPost   = 2;
	cycleType   ='W';
	coarseSolveType = "Julia";
	MG = getMGparam(ComplexF64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.0,0.0);
	shift = 0.2;
	Hparam = HelmholtzParam(Minv,zeros(0),zeros(0),0.0,true,true);
	Ainv = getShiftedLaplacianMultigridSolver(Hparam, MG,shift,"BiCGSTAB",0,true);
else
	Ainv  = getParallelJuliaSolver(ComplexF64,Int64,numCores=16,backend=3);
end

workersFWI = workers();
println(string("The workers that we allocate for FWI are:",workersFWI));


figure(1,figsize = (22,10));
plotModel(m,includeMeshInfo=true,M_regular = Minv,cutPad=pad,limits=[1.5,4.5],figTitle="mref",filename="orig.png");

figure(2,figsize = (22,10));
plotModel(mref,includeMeshInfo=true,M_regular = Minv,cutPad=pad,limits=[1.5,4.5],figTitle="mref",filename="mref.png");


prepareFWIDataFiles(m,Minv,mref,boundsHigh,boundsLow,dataFilenamePrefix,omega,ones(ComplexF64,size(omega)),
									pad,ABLpad,jumpSrc,jumpRcv,offset,workersFWI,maxBatchSize,Ainv,useFilesForFields);


(Q,P,pMis,SourcesSubInd,contDiv,Iact,sback,mref,boundsHigh,boundsLow) =
	setupFWI(m,dataFilenamePrefix,plotting,workersFWI,maxBatchSize,Ainv,SSDFun,useFilesForFields, true);

########################################################################################################
# Setting up the inversion for slowness instead of velocity:
########################################################################################################
function dump(mc,Dc,iter,pInv,PMis,resultsFilename)
	if iter==0
		return;
	end
	fullMc = slowSquaredToVelocity(reshape(Iact*pInv.modelfun(mc)[1] + sback,tuple((pInv.MInv.n)...)))[1];
	Temp = splitext(resultsFilename);
	if iter>0
		Temp = string(Temp[1],iter,Temp[2]);
	else
		Temp = resultsFilename;
	end
	if resultsFilename!=""
		writedlm(Temp,convert(Array{Float16},fullMc));
	end
	if plotting
		figure(888,figsize = (22,10));
		clf();
		filename = splitdir(Temp)[2];
		plotModel(fullMc,includeMeshInfo=true,M_regular = Minv,cutPad=pad,limits=[1.5,4.5],filename=filename,figTitle=filename);
	end
end


#####################################################################################################
# Setting up the inversion for velocity:
#####################################################################################################
mref 		= velocityToSlowSquared(mref)[1];
t    		= copy(boundsLow);
boundsLow 	= velocityToSlowSquared(boundsHigh)[1];
boundsHigh 	= velocityToSlowSquared(t)[1]; t = 0;
modfun 		= identityMod;

########################################################################################################
# Set up Inversion #################################################################################
########################################################################################################

flush(Base.stdout)

GN = "projGN"
maxStep=0.05*maximum(boundsHigh);
regparams = [1.0,1.0,1.0,1e-6];
regfunLow(m,mref,M) 	= wdiffusionReg(m,mref,M,Iact=Iact,C=[]);
regfunHigh(m,mref,M) 	= wFourthOrderSmoothing(m,mref,M,Iact=Iact,C=[]);
if dim==2
	HesPrec=getExactSolveRegularizationPreconditioner();
else
	HesPrec = getSSORCGFourthOrderRegularizationPreconditioner(regparams,Minv,Iact,1.0,1e-8,1000);
end

alpha 	= 1e+2;
pcgTol 	= 1e-1;
maxit 	= 10;

pInv = getInverseParam(Minv,modfun,regfunHigh,alpha,mref[:],boundsLow,boundsHigh,
                         maxStep=maxStep,pcgMaxIter=cgit,pcgTol=pcgTol,
						 minUpdate=1e-3, maxIter = maxit,HesPrec=HesPrec);
mc = copy(mref[:]);


function saveCheckpoint(resultsFilename,mc,Z1,Z2,alpha1,alpha2,pInv,cyc)
	file = matopen(string(splitext(resultsFilename)[1],"_Cyc",cyc,"_checkpoint.mat"), "w");
	write(file,"mc",mc);
	write(file,"Z1",Z1);
	write(file,"Z2",Z2);
	write(file,"alpha1",alpha1);
	write(file,"alpha2",alpha2);
	write(file,"alpha",pInv.alpha);
	write(file,"mref",pInv.mref);
	close(file);
	println("****************************************************************************")
	println("*********************** Saving Checkpoint for cycle ",cyc," ********************")
	println("****************************************************************************")
end

function loadCheckpoint(resultsFilename,cyc)
	file = matopen(string(splitext(resultsFilename)[1],"_Cyc",cyc,"_checkpoint.mat"), "r");
	mc = read(file,"mc");
	Z1 = read(file,"Z1");
	Z2 = read(file,"Z2");
	alpha1 = read(file,"alpha1");
	alpha2 = read(file,"alpha2");
	alpha = read(file,"alpha");
	mref = read(file,"mref");
	close(file);
	return mc,Z1,Z2,alpha1,alpha2,alpha,mref
end


if chosenMethod == FWI || chosenMethod == FWI_SS
	# Standard FWI run
	freqContParams = getFreqContParams(mc, 0, Q,size(P,2), pInv, pMis,
			windowSize, resultsFilename,dump,Iact,sback,
			simSrcDim = simSrcDim);

elseif chosenMethod == FWI_ES_SS || chosenMethod == FWI_ES
	N_nodes = prod(Minv.n.+1);
	nsrc = size(Q,2);
	p = 16;
	Z1 = 2e-4*rand(ComplexF64,(N_nodes, p));
	freqContParams = getFreqContParams(mc, 0, Q,size(P,2), pInv, pMis,
			windowSize, resultsFilename,dump,Iact,sback, Z1=Z1, alpha1=alpha1,
			alpha2Orig=alpha2, stepReg=stepReg,
			simSrcDim = simSrcDim, FWImethod="FWI_ES");
end

for i = 1:freqContSweeps
	freqContParams.cycle = i - 1;
	freqContParams.itersNum = GNiters[i];
	freqContParams.startFrom = freqRanges[i][1];
	freqContParams.endAt = freqRanges[i][2];
	if i > EScycles
		freqContParams.FWImethod = "FWI";
	end
	if regularizations[i] == "low"
		freqContParams.pInv.regularizer = regfunLow;
		freqContParams.updateMref = true;
	else
		freqContParams.pInv.regularizer = regfunHigh;
		freqContParams.updateMref = false;
	end
	mc, = freqCont(freqContParams);
	freqContParams.mc = mc;
end
