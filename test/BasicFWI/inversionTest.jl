include("setupFWItests.jl");


using jInv.InverseSolve
using jInv.LinearSolvers
using jInvVis
using PyPlot
close("all")
figure(20)
imshow(m');colorbar();

figure(21);
imshow(reshape(gamma,(nx,nz))')

Dp,pFor = getData(vec(m),pFor,false)
nfreq = length(omega);

Dobs = fetch(Dp);
Wd = 1.0./(Dobs .+ 1e-2*maximum(abs.(Dobs[:])));


# Wd   = Array{Array{Float64,2}}(undef,length(omega))
# Dobs = Array{Array{Float64,2}}(undef,length(omega))

# for k=1:nfreq
	# Dobs[k] = fetch(Dp[k]);
	# Wd[k] = 1.0./(Dobs[k] .+ 1e-2*maximum(abs(Dobs[k][:])));
	# figure(k);
	# imshow(Dobs[k])
# end


#min_m misfit(W_d*(getData(modelFun(m)), Dobs))

#1,10,100,10^4
#s = exp(m)

mref = 3.0*ones(nx,nz);
mref = 1.0./(mref.^2); # convert to slowness squared

N = prod(Mr.n);
Iact = sparse(I,N,N);
mback   = zeros(Float64,N);

########################################################################################################
##### Set up remote workers ############################################################################
########################################################################################################

Ainv = getJuliaSolver(); 
## Choose the workers for FWI (here, its a single worker)
misfun = SSDFun;
pMis = getMisfitParam(pFor, Wd, Dobs, misfun);

boundsHigh = 0.12*ones(Float32,N);
boundsLow = 0.035*ones(Float32,N);

maxStep				=0.05*maximum(boundsHigh);

#regparams 			= [1.0,1.0,1.0,1e-5];
cgit 				= 8; 
alpha 				= 1e+2;
pcgTol 				= 1e-1;
maxit 				= 30;
HesPrec 			= getExactSolveRegularizationPreconditioner();
regfun(m,mref,M) 	= wdiffusionReg(m,mref,M,Iact=Iact,C=[]);


pInv = getInverseParam(Mr,identityMod,regfun,alpha,mref[:],boundsLow,boundsHigh,
                         maxStep=maxStep,pcgMaxIter=cgit,pcgTol=pcgTol,
						 minUpdate=1e-3, maxIter = maxit,HesPrec=HesPrec);

						 
plotting = true;			 
function plotIntermediateResults(mc,Dc,iter,pInv,PMis,resultsFilename="")
	# Models are usually shown in velocity.
	fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
	#fullMc = reshape(pInv.modelfun(mc)[1],tuple((pInv.MInv.n)...));
	if plotting
		close(888);
		figure(888);
		#plotModel(fullMc,false,[],0,[1.5,4.8]);
		imshow(fullMc');colorbar();
		pause(1.0)
	end
end
						 
# Run one sweep of a frequency continuation procedure.
# mc,Dc,flag,His = freqCont(copy(mref[:]), pInv, pMis,contDiv, 4, "",plotIntermediateResults,"Joint",1,1,"projGN");

mc, = projGN(copy(mref[:]),pInv,pMis,dumpResults=plotIntermediateResults,out=2);
a = 1;
