
# get a small mesh
domain = [0 2.0 0 2.2 0 1.2]
n      = [12;8; 7]
M      = getRegularMesh(domain,n)

# sources / receivers on top edge
idx    = reshape(collect(1:prod(n.+1)),tuple(n.+1...))
ib     = idx[:,1,:];

n_nodes = prod(n.+1);
Q      = SparseMatrixCSC(1.0I, n_nodes, n_nodes)
Q      = Q[:,vec(ib)]
R      = copy(Q)

# get param without parallelizaion
pFor = getEikonalInvParam(M,Q,R,true)
m0   = 0.2*rand(Float64,tuple(n.+1...)) .+ 1.0
dho, = getData(vec(m0),pFor)
pFor.HO = false;
dlo, = getData(vec(m0),pFor)
@test norm(dho-dlo)/norm(dho) < 0.05

# parallelize over sources
pForp,continuationDivision,SourcesSubInd = getEikonalInvParam(M,Q,R,true,4)
dphor, = getData(vec(m0),pForp)
dpho = zeros(size(dho))
for k=1:4
	dpho[:,SourcesSubInd[k]] = fetch(dphor[k])
end
pForp,continuationDivision,SourcesSubInd = getEikonalInvParam(M,Q,R,false,4)
dplor, = getData(vec(m0),pForp)
dplo = zeros(size(dho))
for k=1:4
	dplo[:,SourcesSubInd[k]] = fetch(dplor[k])
end

@test norm(dpho-dho)/norm(dho) < 1e-12
@test norm(dplo-dlo)/norm(dlo) < 1e-12

