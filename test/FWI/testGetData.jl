




# get a small mesh
domain = [0 2.0 0 2.2]
n      = [12;8;]
M      = getRegularMesh(domain,n)
omega  = 0.1; 

n_nodes = n.+1;
nn = prod(n_nodes);
# sources / receivers on top edge
idx    = reshape(collect(1:nn),tuple(n_nodes...))
ib     = idx[:,1];
Q      = SparseMatrixCSC(1.0I, nn, nn);
Q      = Q[:,vec(ib)]
R      = copy(Q)

# get param without parallelizaion
ABLpad = 2;
gamma = getABL(M,true,ones(Int64,M.dim)*ABLpad,8*pi);
attenuation = 0.01*4*pi;
gamma .+= attenuation; # adding Attenuation.
gamma = gamma[:];
Ainv = getJuliaSolver();

pFor = getFWIparam(omega, one(ComplexF64),gamma,Q,R,M,Ainv,[workers()[1]])[1];

m0   = rand(Float64,tuple(n.+1...)).+1.0
dobs, = getData(vec(m0),fetch(pFor[1]));


# parallelize over sources
pForp,continuationDivision,SourcesSubInd = getFWIparam(omega, one(ComplexF64),gamma,Q,R,M,Ainv,workers())
dobsRF, = getData(vec(m0),pForp)
Dobs = zeros(eltype(dobs),size(dobs));
for k=1:length(dobsRF)
	dobsk = fetch(dobsRF[k]);
	Dobs[:,SourcesSubInd[k]] = dobsk;
end

@test norm(Dobs-dobs)/norm(dobs) < 1e-8



