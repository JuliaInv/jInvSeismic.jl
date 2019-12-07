using DelimitedFiles
using jInvVis
using PyPlot

figure();
m = DelimitedFiles.readdlm("ES_FWI_FC5_5_GN1.dat");
# m = DelimitedFiles.readdlm("Ns1_GN5_FC1");
# ns = maximum(m[:])
# ind = findfirst(a->a==ns, m)
# m[ind] = 0.0;
# m = reshape(m, (67,34));

imshow(m'); colorbar();
