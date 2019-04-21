include("setupInversionExample.jl");

@everywhere begin
using jInv.InverseSolve
using jInv.LinearSolvers
end

using jInvVis
using PyPlot

close("all")
tstart = time_ns();

function plotInputData()
	figure(20)
	imshow(m');colorbar();

	figure(21);
	imshow(reshape(gamma,(nx,nz))');
end

plotInputData();
Dobs, Wd = solveForwardProblem(m, pForp, omega, nrec, nsrc, nfreq);

function plotModel(model)
	close(888);
	figure(888);
	imshow(model);colorbar();
	pause(1.0);
end
mc, Dc, pInv, Iact, mback = solveInverseProblem(pForp, Dobs, Wd, nfreq, nx, nz, Mr, true, plotModel);



fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
println("Model error:");
println(norm(fullMc.-m));
tend = time_ns();
println("Runtime:");
println((tend - tstart)/1.0e9);

#Plot residuals
figure(22);
imshow((abs.(m.-fullMc))'); colorbar();
