using jInvVis
using PyPlot
using LinearAlgebra
import jInv.Utils.clear!

include("setupInversionExample.jl");

close("all")

function startExample()
	tstart = time_ns();

	function plotInputData()
		figure(20)
		imshow(m');colorbar();

		figure(21);
		imshow(reshape(gamma,tuple(size(m)...))');
	end
	plotInputData();

	Dobs, Wd = solveForwardProblemExtendedSources(m, pForp, omega, nfreq);

	# plot observed data
	for i=1:nfreq
		figure(i);
		imshow(Dobs[i][:,:]);
	end

	function plotModelResult(model)
		close(888);
		figure(888);
		imshow(model);colorbar();
		pause(1.0);
	end
	nx = size(m)[1];
	nz = size(m)[2];

	figure(22);
	imshow(mref'); colorbar();

	mc, Dc, pInv, Iact, mback, pMis = solveInverseProblemExtendedSources(pForp, Dobs, Wd, nfreq, nx, nz, mref,
								Mr, 0.5, 0.035,"ES_FWI.dat", true, plotModelResult);
	# Show results
	fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
	println("Model error:");
	println(norm(fullMc.-m));
	tend = time_ns();
	println("Runtime:");
	println((tend - tstart)/1.0e9);

	#Plot residuals
	figure(23);
	imshow((abs.(m.-fullMc))'); colorbar();
end

startExample()
