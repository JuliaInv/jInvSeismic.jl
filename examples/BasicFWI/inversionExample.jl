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
	imshow(reshape(gamma,tuple(size(m)...))');
end
println(typeof(pFor));
plotInputData();
Dobs, Wd = solveForwardProblem(m, pForp, omega, nrec, nsrc, nfreq);

# DelimitedFiles.writedlm("Dobs",convert(Array{Float64,3},Dobs));
# DelimitedFiles.writedlm("Wd",convert(Array{Float64,3},Wd));

# Dobs = DelimitedFiles.readdlm("Dobs")
# Wd = DelimitedFiles.readdlm("Wd")
# println(size(vec(Wd[1][:,1])));
# etaM = 2 .*diagm(0 => vec(Wd[1][:,1])).*diagm(0 => vec(Wd[1][:,1]));
# println(size(etaM))
for i=1:nfreq
	figure(i);

	imshow(Dobs[i][:,:]);
end

TEmat = rand([-1,1],(nsrc,5));
pForpTE = getFWIparam(omega,gamma,Q,P,Mr,TEmat,true)
bla, = getData(vec(m),pForpTE)
for k=1:length(bla)
	wait(bla[k]);
end
blaobs = Array{Array}(undef, nfreq);
for k=1:length(bla)
	blaobs[k] = fetch(bla[k]);
end
println(size(blaobs))
println(size(blaobs[1]))
figure(99);
imshow(blaobs[1][:,:]);
#
# function plotModelResult(model)
# 	close(888);
# 	figure(888);
# 	imshow(model);colorbar();
# 	pause(1.0);
# end
# nx = size(m)[1];
# nz = size(m)[2];
#
# figure(22);
# imshow(mref'); colorbar();
#
# mc, Dc, pInv, Iact, mback = solveInverseProblem(pForp, Dobs, Wd, nfreq, nx, nz, mref,
# 												Mr, 0.5, 0.035,"FWI.dat", true, plotModelResult);
#
# #Show results
# fullMc = reshape(Iact*pInv.modelfun(mc)[1] + mback,tuple((pInv.MInv.n)...));
# println("Model error:");
# println(norm(fullMc.-m));
# tend = time_ns();
# println("Runtime:");
# println((tend - tstart)/1.0e9);
#
# #Plot residuals
# figure(23);
# imshow((abs.(m.-fullMc))'); colorbar();
