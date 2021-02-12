export getData

import jInv.ForwardShare.getData
function getData(m,pFor::BasicFWIparam,doClear::Bool=false)
    tstart = time_ns();
    # extract pointers
    Mesh  = pFor.Mesh
    omega = pFor.omega
    gamma = pFor.gamma
    Q     = pFor.Sources
    P     = pFor.Receivers

    nrec  = size(P,2)
    nsrc  = size(Q,2)
    nfreq = length(omega)

    # allocate space for data and fields
    D  = zeros(nrec,nsrc,nfreq)
    U  = zeros(ComplexF64,prod(Mesh.n.+1),nsrc,nfreq)
    LU = Array{Any}(undef, nfreq)
    HinvtPs = Array{Any}(undef, nfreq)
    for i=1:length(omega)
        t1 = time_ns();
        H = getHelmholtzOperator(m,gamma,omega[i],Mesh)
        t2 = time_ns();
        LU[i] = lu(H)
        H = nothing
        for k=1:nsrc
            U[:,k,i] = LU[i]\Vector(Q[:,k])
            D[:,k,i] = real(P'*U[:,k,i])
        end
    end
    pFor.Ainv   = LU
    pFor.Fields = U # will need to save to file
    tend = time_ns();
    println("Runtime of getData:");
    println((tend - tstart)/1.0e9);

    return D ,pFor
end
