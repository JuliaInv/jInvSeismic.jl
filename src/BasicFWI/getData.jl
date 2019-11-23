export getData

import jInv.ForwardShare.getData
function getData(m,pFor::BasicFWIparam,doClear::Bool=false)
    # println("starting GetData5");
    tstart = time_ns();
    # extract pointers
    Mesh  = pFor.Mesh
    omega = pFor.omega
    gamma = pFor.gamma
    Q     = pFor.Sources
    P     = pFor.Receivers
    # Qo = pFor.originalSources

    nrec  = size(P,2)
    nsrc  = size(Q,2)
    nfreq = length(omega)

    # allocate space for data and fields
    D  = zeros(nrec,nsrc,nfreq)
    U  = zeros(ComplexF64,prod(Mesh.n.+1),nsrc,nfreq)
    # H = zeros(ComplexF64,prod(Mesh.n.+1), prod(Mesh.n.+1),nfreq)
    # store factorizations
    LU = Array{Any}(undef, nfreq)
    HinvtPs = Array{Any}(undef, nfreq)
    for i=1:length(omega)
        t1 = time_ns();
        H = getHelmholtzOperator(m,gamma,omega[i],Mesh)
        # println("H: ", H);
        # println("HINV: ", inv(H));
        t2 = time_ns();
        # println("Runtime of helmholtz:");
        # println((t1 - t2)/1.0e9);
        LU[i] = lu(H)
        H = nothing
        for k=1:nsrc
            # println("AA")
            # println(typeof(P))
            # println(size(LU[i]))
            # HinvtPs[i] = (LU[i])'\Matrix(P)
            U[:,k,i] = LU[i]\Vector(Q[:,k])
            # println("change 555555555555555555555555");
            # U[:,k,i] = (HinvtPs[i])' * Vector(Q[:,k])
            D[:,k,i] = real(P'*U[:,k,i])
            # Ucur = LU[i]\Vector(Q[:,k]);
            # D[:,k,i] = real(P'*Ucur);
        end
    end
    pFor.Ainv   = LU
    # pFor.HinvtPs = HinvtPs;
    # pFor.Helmholtz = H
    pFor.Fields = U # will need to save to file
    # writedlm(string(replace(string("U",omega[1]), "." => "_"),".mat"), U)
    tend = time_ns();
    println("Runtime of getData:");
    println((tend - tstart)/1.0e9);

    #clear some memory on workers
    # LU = nothing
    # U = nothing
    # Q = nothing
    # P = nothing
    # omega = nothing
    # gamma = nothing
    # Mesh = nothing

    return D ,pFor
end
