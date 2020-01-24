using KrylovMethods
using JLD
using jInv.LinearSolvers
using Statistics
m = 67;
p = 10;
n = 34;

q = 58;

nsrc = 15

Z1 = 10 * rand(ComplexF64,(m*n,p)); #Initial guess
Z2 = 10 * rand(ComplexF64,(p,nsrc)); #Initial guess

p = 10


alpha = 1e-2;

PH = load("SavedVals2.jld", "hinvp")
DOBS = load("SavedVals2.jld", "dobs")[:,:,1]
WD = load("SavedVals2.jld", "wd")[:,:,1]
SRC = load("SavedVals2.jld", "sources")

function misfitCalc()
	sum = 0;
	sum += (mean(WD)^2) .* norm(PH' * (SRC + Z1 * Z2) -DOBS)^2;

	sum	+= alpha * norm(Z1)^2 + alpha * norm(Z2)^2;
	return sum;
end



println("misfit at start:: ", misfitCalc())
rhs= (mean(WD)^2) .* Z1' * PH * (-PH' * SRC + DOBS);
lhs = zeros(ComplexF64, (p,p));
lhs += (mean(WD)^2) .* Z1' * PH * PH' * Z1;


lhs += alpha * I;

Z2 = lhs\rhs;


println("misfit at Z2:: ", misfitCalc())

function multOP(R,HinvP)
	return HinvP' * R * Z2;
end

function multOPT(R,HinvP)
	return HinvP * R * Z2';
end

function multAll(x)
	sum = zeros(ComplexF64, (m*n, p));

	sum += (mean(WD)^2) .* multOPT(multOP(x, PH), PH);

	sum += alpha * x;
	return sum;
end

rhs = zeros(ComplexF64, (m*n, p));
rhs += (mean(WD)^2).*multOPT(-PH' * SRC + DOBS, PH);


Z1 = KrylovMethods.blockBiCGSTB(x-> multAll(x), rhs)[1];

println("misfit at Z1:: ", misfitCalc())
