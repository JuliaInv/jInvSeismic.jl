include("setupInversionTest.jl");

@everywhere begin
using jInv.InverseSolve
using jInv.LinearSolvers
end

using Test

println("Running inversion test");

Dobs, Wd = solveForwardProblem(m ,pForp, omega, nrec, nsrc, nfreq);
mc, Dc, pInv, Iact, mback = solveInverseProblem(pForp, Dobs, Wd, nfreq, nx, nz, Mr);

fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
println("Model error:");
println(norm(fullMc.-m));
@test norm(fullMc.-m) < 1.4

for k=1:length(Dc)
	wait(Dc[k]);
end
Dinv = Array{Array}(undef, nfreq);
for k=1:length(Dc)
	Dinv[k] = fetch(Dc[k]);
end
println("Data error:");
println(norm(Dobs.-Dinv));
@test norm(Dobs.-Dinv) < 5.4
