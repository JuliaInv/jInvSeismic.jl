using jInvSeismic.FWI
using jInvSeismic.Utils
using Helmholtz
using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using Multigrid.ParallelJuliaSolver
using Distributed

using DelimitedFiles

if nworkers() == 1
	addprocs(16);
elseif nworkers() < 16
	addprocs(16 - nworkers());
end

@everywhere begin
using jInv.InverseSolve
using jInv.LinearSolvers
using jInvSeismic.FWI
using jInv.Mesh
using jInv.Utils
using DelimitedFiles
using jInv.ForwardShare
using KrylovMethods
end

dim     = 2;
pad     = 30;
newSize = [600,300];

(m,Mr,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMref("examples","SEGmodel2Dsalt.dat",dim,pad,[0.0,13.5,0.0,4.2],newSize,1.752,2.7);
m = 1 ./ (m.^2);
mref = 1 ./ (mref.^2);

# attenuation for BC
padx = 34; padz = 34
a    = 2.0;
xc = getCellCenteredGrid(Mr);
gamma = getHelmholtzABL(Mr,true,[padx;padz],a);
# gamma = Helmholtz.getABL(Mr,true,[padx;padz],a);
println("Size gamma: ", size(gamma))
println("Size Mr: " , Mr.n)
println("Size m: " , size(m))
# parameters for the Helmholtz (units in km)
h = Mr.h;
n = Mr.n;
# omega = 0.3*2*pi*[2.0;5.0; 10.0;20.0;25.0]
omega = 2*pi*[2.0;2.5;3.5;4.5;6.0;]
nfreq = length(omega)
# generate sources
q = zeros(tuple(n.+1...)); q[padx+1:4:end-padx-1,1] .= 1e4
print(size(q))
Q = sdiag(vec(q))
print(size(Q))
Q = Q[:,(LinearIndices(sum(Q,dims=2) .!= 0))[findall(sum(Q,dims=2) .!= 0)]]
print(size(Q))
nsrc = size(Q,2)
# receivers
p = zeros(tuple(n.+1...)); p[padx+1:end-padx-1,1] .= 1
P = sdiag(vec(p))
P = P[:,(LinearIndices(sum(P,dims=2) .!= 0))[findall(sum(P,dims=2) .!= 0)]]
nrec = size(P,2)
Ainv  = getParallelJuliaSolver(ComplexF64,Int64,numCores=4,backend=1);

(pForp,contDivFWI,SourcesSubIndFWI)  = getFWIparam(omega,ones(ComplexF64,size(omega)),vec(gamma),Q,P,Mr,Ainv,workers())
# pForp = getBasicFWIparam(omega,gamma,Q,P,Mr,true)

# P = nothing
# Q = nothing
# inversion mesh and forward mesh are the same here
M2Mp = ones(length(pForp))
