export getData

function getData(m,pFor::FWIparam,doClear::Bool=false)

    # extract pointers
    M       	= pFor.Mesh
    omega   	= pFor.omega
	wavelet 	= pFor.WaveletCoef;
    gamma   	= pFor.gamma
    Q       	= pFor.Sources
    P       	= pFor.Receivers
	Ainv    	= pFor.ForwardSolver;
	batchSize 	= pFor.forwardSolveBatchSize;
	select  	= pFor.sourceSelection;

    nrec  		= size(P,2)
    nsrc  		= size(Q,2)

	An2cc = getNodalAverageMatrix(M);

    m = An2cc'*m;
	gamma = An2cc'*gamma;

	# allocate space for data and fields
	n_nodes = prod(M.n.+1);
	# ALL AT ONCE DIRECT CODE
	H = GetHelmholtzOperator(M,m,omega, gamma, true,useSommerfeldBC);

	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		Ainv.helmParam = HelmholtzParam(M,gamma,m,omega,true,useSommerfeldBC);
		H = H + GetHelmholtzShiftOP(m, omega,Ainv.shift[1]);
		H = sparse(H');
		# H is actually shifted laplacian now...
		Ainv.MG.relativeTol *= 1e-4;
	end


	if select==[]
		Qs = Q*wavelet;
	else
		Qs = Q[:,select]*wavelet;
	end

	nsrc 		= size(Qs,2);

	if batchSize > nsrc
		batchSize = nsrc;
	end


	Fields = [];

	if doClear==false
		if pFor.useFilesForFields
			tfilename = getFieldsFileName(omega);
			tfile     = matopen(tfilename, "w");
		else
			Fields    = zeros(FieldsType,n_nodes   ,nsrc);
		end
	end

	numBatches 	= ceil(Int64,nsrc/batchSize);
	D 			= zeros(FieldsType,nrec,nsrc);
	U 			= zeros(FieldsType,n_nodes,batchSize);

	Ainv.doClear = 1;
	for k_batch = 1:numBatches
		println("handling batch ",k_batch," out of ",numBatches);
		batchIdxs = (k_batch-1)*batchSize + 1 : min(k_batch*batchSize,nsrc);
		if length(length(batchIdxs))==batchSize
			U[:] = convert(Array{FieldsType},Matrix(Qs[:,batchIdxs]));
		else
			U = convert(Array{FieldsType},Matrix(Qs[:,batchIdxs]));
		end

		@time begin
			ts = time_ns();
			U,Ainv = solveLinearSystem(H,U,Ainv,0)
			es = time_ns();
			println("Runtime of Solve LS: ", (es - ts) / 10e9);
		end

		Ainv.doClear = 0;
		D[:,batchIdxs] = (P'*U);

		if doClear==false
			if pFor.useFilesForFields
				write(tfile,string("Ubatch_",k_batch),convert(Array{ComplexF64},U));
			else
				Fields[:,batchIdxs] = U;
			end
		end
	end

	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		Ainv.MG.relativeTol *= 1e+4;
	end

	pFor.ForwardSolver = Ainv;

	if doClear==false
		if pFor.useFilesForFields
			close(tfile);
		else
			pFor.Fields = Fields;
		end
	end

	if doClear
		clear!(pFor);
	elseif isa(Ainv,ShiftedLaplacianMultigridSolver)
		clear!(Ainv.MG); 
	end
    return D,pFor
end
