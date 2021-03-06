if nworkers()<=1
	addprocs(4);
end

using jInv.Mesh
using jInv.Utils
using jInv.InverseSolve
using jInvSeismic.EikonalInv
using jInvSeismic.Utils
using MAT
using DelimitedFiles
using Statistics

#############################################################################################################
modelDir = "../examples";

dataDir = pwd();
include("../../examples/TravelTimeTomography/drivers/prepareTravelTimeDataFiles.jl");
include("../../examples/TravelTimeTomography/drivers/setupTravelTimeTomography.jl");

########################################################################################################
######################################## for SEG 256X512 ###############################################
#######################################################################################################
dim     = 2;
pad     = 1;
jump    = 32;
offset  = 256;
(m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateNodalMeshMref(modelDir,"SEGmodel2Dsalt.dat",dim,pad,[0.0,13.5,0.0,4.2],[64,32]);
dataFilenamePrefix = string(dataDir,"/DATA_SEG",tuple((Minv.n.+1)...));


prepareTravelTimeDataFiles(m,Minv,mref,boundsHigh,boundsLow,dataFilenamePrefix,pad,jump,offset);
resultsFilenamePrefix = "";
(Q,P,pMisRFs,SourcesSubInd,contDiv,Iact,sback,mref,boundsHigh,boundsLow,resultsFilename) = setupTravelTimeTomography(m,dataFilenamePrefix, resultsFilenamePrefix);


########################################################################################################
##### Set up Inversion #################################################################################
########################################################################################################
maxStep=0.1*maximum(boundsHigh);
a = minimum(boundsLow);
b = maximum(boundsHigh);
function modfun(m)
	bm,dbm = getBoundModel(m,a,b);
	bs,dbs = velocityToSlowSquared(bm);
	dsdm = dbm*dbs;
	return bs,dsdm;
end
mref = getBoundModelInv(mref,a,b);
boundsHigh = boundsHigh*10000000.0;
boundsLow = -boundsLow*100000000.0

cgit  = 2;
maxit = 1;
alpha = 1e+0;
pcgTol = 1e-1;

HesPrec=getSSORCGRegularizationPreconditioner(1.0,1e-5,2)

################################################# GIT VERSION OF JINV #################################################

regparams = [1.0,1.0,1.0,1e-2];
regfun(m, mref, M) = wdiffusionRegNodal(m, mref, M, Iact=Iact, C = regparams);
	
pInv = getInverseParam(Minv,modfun,regfun,alpha,mref[:],boundsLow,boundsHigh,
                     maxStep=maxStep,pcgMaxIter=cgit,pcgTol=pcgTol,
					 minUpdate=1e-3, maxIter = maxit,HesPrec=HesPrec);
					 
mc,Dc,flag = projGNCG(mref[:],pInv,pMisRFs,indCredit = []);

clear!(pMisRFs);
#############################################################################################
rm("DATA_SEG(66, 33)_travelTime.dat");
rm("DATA_SEG(66, 33)_rcvMap.dat");
rm("DATA_SEG(66, 33)_srcMap.dat");
rm("DATA_SEG(66, 33)_PARAM.mat");
rm("jInv.out");


