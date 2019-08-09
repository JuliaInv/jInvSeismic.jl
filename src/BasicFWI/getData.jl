export getData

import jInv.ForwardShare.getData
function getData(m,pFor::BasicFWIparam,doClear::Bool=false)
    println("starting GetData5");
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
    for i=1:length(omega)
        t1 = time_ns();
        H = getHelmholtzOperator(m,gamma,omega[i],Mesh)
        t2 = time_ns();
        println("Runtime of helmholtz:");
        println((t1 - t2)/1.0e9);
        println(size(H))
        LU[i] = lu(H)
        println(size(LU[i]));
        for k=1:nsrc
            println(size(Vector(Q[:,k])));
            U[:,k,i] = LU[i]\Vector(Q[:,k])
            D[:,k,i] = real(P'*U[:,k,i])
            # Ucur = LU[i]\Vector(Q[:,k]);
            # D[:,k,i] = real(P'*Ucur);
        end
    end
    pFor.Ainv   = LU
    # pFor.Helmholtz = H
    pFor.Fields = U
    tend = time_ns();
    println("Runtime of getData:");
    println((tend - tstart)/1.0e9);
    return D ,pFor
end
