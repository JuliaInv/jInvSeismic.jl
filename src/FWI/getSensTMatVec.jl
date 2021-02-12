export getSensTMatVec

function getSensTMatVec(v::Vector,m::Vector,pFor::FWIparam)

    # extract pointers
    M  				= pFor.Mesh
    omega 			= pFor.omega
    gamma 			= pFor.gamma
    Q     			= pFor.Sources
    P     			= pFor.Receivers
    Ainv			= pFor.ForwardSolver

	batchSize 		= pFor.forwardSolveBatchSize;
	select  		= pFor.sourceSelection;

	nsrc = size(Q,2); nrec = size(P,2);

	if length(select) > 0
		nsrc = length(select);
	end

	if batchSize > nsrc
		batchSize = nsrc;
	end

	numBatches 	= ceil(Int64,nsrc/batchSize);
	An2cc = getNodalAverageMatrix(M);
	m_nodal = An2cc'*m;
	gamma_nodal = An2cc'*gamma;
	n_nodal = prod(M.n.+1);
	H = spzeros(ComplexF64,n_nodal,n_nodal);
	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		H = GetHelmholtzOperator(M,m_nodal,omega, gamma_nodal, true,useSommerfeldBC);
		Ainv.helmParam = HelmholtzParam(M,gamma_nodal,m_nodal,omega,true,useSommerfeldBC);
		H = H + GetHelmholtzShiftOP(m_nodal, omega,Ainv.shift[1]);
		H = sparse(H');
		# H is actually shifted laplacian now...
	elseif isa(Ainv,JuliaSolver)
		H = GetHelmholtzOperator(M,m_nodal,omega, gamma_nodal, true,useSommerfeldBC);
	end

	JTv = zeros(Float64,prod(M.n))
	Vdatashape = reshape(v,nrec,nsrc);

	for k_batch = 1:numBatches
		batchIdxs = (k_batch-1)*batchSize + 1 : min(k_batch*batchSize,nsrc);
		V = P*Vdatashape[:,batchIdxs];
		V,Ainv = solveLinearSystem(H,V,Ainv,1);
		if pFor.useFilesForFields
			filename = getFieldsFileName(omega);
			file     = matopen(filename);
			U        = read(file,string("Ubatch_",k_batch));
			close(file);
		else
			U = pFor.Fields[:,batchIdxs];
		end
		# Original code:  V = conj((1.0.-1im*vec(gamma./omega)).*U).*V;
		V = An2cc*(conj(U).*(1.0.+1im*vec(gamma_nodal./omega)).*V);
		JTv .+= omega^2*vec(real(sum(V,dims=2)));
	end

	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		clear!(Ainv.MG);
	end

    return JTv
end
