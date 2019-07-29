using jInvSeismic.BasicFWI
using jInvSeismic.Utils
using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using Distributed

using DelimitedFiles

if nworkers() == 1
	addprocs(6);
elseif nworkers() < 6
	addprocs(6 - nworkers());
end


println("IM here")
@everywhere begin
	using jInvSeismic.BasicFWI
	using jInv.Mesh
	using jInv.Utils
	using jInv.ForwardShare
end



# setup model and attenuation function
nx = 256; nz = 128
domain = [0.0,10.0,0.0,4.0]
Mr = getRegularMesh(domain,[nx,nz])

# the slowness model
m = 3.0*ones(nx,nz);
m[30:80,20:40] .= 5.0; # velocity...
m = 1.0./(m.^2); # convert to slowness squared


# m = readdlm("SEGmodel2Dsalt.dat");
# m = m*1e-3;
# m = m';
# m = 1.0./m^2



# attenuation for BC
padx = 20; padz = 20
a    = 2.0;
xc = getCellCenteredGrid(Mr)
gamma = getABL(Mr,true,[padx;padz],a);

# parameters for the Helmholtz (units in km)
h = Mr.h;
n = [nx; nz;]
omega = 2*pi*[0.5;1.0;2.0;3.0;4.0;5.0;6.0]
nfreq = length(omega)

# generate sources
q = zeros(tuple(n.+1...)); q[padx+1:4:end-padx-1,1] .= 1e4
Q = sdiag(vec(q))
Q = Q[:,(LinearIndices(sum(Q,dims=2) .!= 0))[findall(sum(Q,dims=2) .!= 0)]]
nsrc = size(Q,2)

# receivers
p = zeros(tuple(n.+1...)); p[padx+1:end-padx-1,1] .= 1
P = sdiag(vec(p))
P = P[:,(LinearIndices(sum(P,dims=2) .!= 0))[findall(sum(P,dims=2) .!= 0)]]
nrec = size(P,2)

pFor  = getFWIparam(omega,gamma,Q,P,Mr)
pForp = getFWIparam(omega,gamma,Q,P,Mr,true)

# inversion mesh and forward mesh are the same here
M2Mp = ones(length(pForp))
