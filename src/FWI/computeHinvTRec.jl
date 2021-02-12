export computeHinvTRec

function computeHinvTRec(pMis::Array{RemoteChannel})
HinvTRec = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
	for k=1:length(pMis)
		@async begin
			HinvTRec[k] = remotecall_fetch(computeHinvTRec,pMis[k].where,pMis[k]);
		end
	end
end
return HinvTRec;
end

function calculateShiftedH(pMis, m)
	pFor = pMis.pFor;
	M       	= pFor.Mesh
	omega   	= pFor.omega
	wavelet 	= pFor.WaveletCoef;
	gamma   	= pFor.gamma
	Q       	= pFor.Sources
	P       	= pFor.Receivers
	Ainv    	= pFor.ForwardSolver;
	batchSize 	= pFor.forwardSolveBatchSize;
	select  	= pFor.sourceSelection;
	An2cc = getNodalAverageMatrix(M);

	m, = pMis.modelfun(m)
	m = interpGlobalToLocal(m,pMis.gloc.PForInv,pMis.gloc.sigmaBackground);

	m = An2cc'*m;
	gamma = An2cc'*gamma;

	H = GetHelmholtzOperator(M,m,omega, gamma, true,useSommerfeldBC);
	Ainv.helmParam = HelmholtzParam(M,gamma,m,omega,true,useSommerfeldBC);
	H = H + GetHelmholtzShiftOP(m, omega,Ainv.shift[1]);
	H = sparse(H');

	return H;
end

function computeHinvTRec(pMisRF::RemoteChannel)
pMis = fetch(pMisRF)
HinvTP, = solveLinearSystem(spzeros(ComplexF64,0,0),complex(Matrix(pMis.pFor.Receivers)),pMis.pFor.ForwardSolver,1);
return HinvTP;
end


function computeHinvTRec(pMisRF::RemoteChannel, m)
pMis = fetch(pMisRF)
H = calculateShiftedH(pMis, m)
HinvTP, = solveLinearSystem(H,complex(Matrix(pMis.pFor.Receivers)),pMis.pFor.ForwardSolver,1);
return HinvTP;
end

function computeHinvTRec(pMisRF::RemoteChannel, x, m, doTranspose)
pMis = fetch(pMisRF)
if doTranspose == 0
	HinvTP, = solveLinearSystem(spzeros(ComplexF64,0,0),x,pMis.pFor.ForwardSolver,doTranspose);
	return complex(pMis.pFor.Receivers' * HinvTP);
else
	HinvTP, = solveLinearSystem(spzeros(ComplexF64,0,0),complex(pMis.pFor.Receivers * x),pMis.pFor.ForwardSolver,doTranspose);
	return  HinvTP;
end
end

function computeHinvTRecX(pMis::Array{RemoteChannel}, x, m, doTranspose=1)
HinvTRec = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
		for k=1:length(pMis)
	@async begin
			HinvTRec[k] = remotecall_fetch(computeHinvTRec,pMis[k].where,pMis[k],x, m,doTranspose);
		end
	end
end
return HinvTRec;
end

function computeHinvTRecXarr(pMis::Array{RemoteChannel}, x, m, doTranspose=1)
HinvTRec = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
	for k=1:length(pMis)
	@async begin
			HinvTRec[k] = remotecall_fetch(computeHinvTRec,pMis[k].where,pMis[k],x[k], m,doTranspose);
		end
	end
end
return HinvTRec;
end
