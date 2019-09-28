import jInv.ForwardShare.getSensTMatVec


function getSensTMatVec(v::Vector,m::Vector,pFor::BasicFWIparam)

    # extract pointers
    Mesh  = pFor.Mesh
    omega = pFor.omega
    gamma = pFor.gamma
    Q     = pFor.Sources
    P     = pFor.Receivers
    LU    = pFor.Ainv
    U     = pFor.Fields


    nsrc = size(Q,2); nfreq = length(omega); nrec = size(P,2)

    # reshape v by receivers x sources x frequencies
    v    = reshape(v,nrec,nsrc,nfreq)

    # allocate space for result
    JTv  = zeros(length(m))
    # derivative of mass matrix
    An2cc = getNodalAverageMatrix(Mesh)
    dM    = repeat(1 .- 1im*vec(gamma),1,nsrc)
    for i=1:nfreq
        tf1=time_ns();
        Lam  = LU[i]\(P*v[:,:,i])
        # println(size(An2cc))
        # println(size(Lam))
        # println(size(U[:,:,i]))
        # println(size(dM))
        # println(size(omega[i]))
        JTvi =  omega[i]^2*dM.*(An2cc*(U[:,:,i].*Lam))
        JTv +=  sum(real(JTvi),dims=2)
        tf2=time_ns();
        # println("Runtime of sensT freq:");
        # println((tf1 - tf2)/1.0e9);
    end
    return vec(JTv)
end

function getSensTMatVecZs(v::Vector,m::Vector,pFor::BasicFWIparam)

        # extract pointers
        Mesh  = pFor.Mesh
        omega = pFor.omega
        gamma = pFor.gamma
        Q     = pFor.Sources
        P     = pFor.Receivers
        LU    = pFor.Ainv
        U     = pFor.Fields


        nsrc = size(Q,2); nfreq = length(omega); nrec = size(P,2)

        # reshape v by receivers x sources x frequencies
        v    = reshape(v,nrec,nsrc,nfreq)

        # allocate space for result
        JTv  = zeros(length(m))
        # derivative of mass matrix
        An2cc = getNodalAverageMatrix(Mesh)
        dM    = repeat(1 .- 1im*vec(gamma),1,1)
        for i=1:nfreq
            tf1 = time_ns();
            for s=1:nsrc
                t1 = time_ns();
                Lam  = LU[i]\(P*v[:,s,i])
                JTv += omega[i]^2*dM.*(An2cc*((LU[i]\Vector(Q[:,s])).*Lam))[:,1];
                t2 = time_ns();
                # println("Runtime of sensT iteration:");
                # println((t1 - t2)/1.0e9);
            end
            tf2=time_ns();
            # println("Runtime of sensT freq:");
            # println((tf1 - tf2)/1.0e9);

        end
        return vec(JTv)
end
