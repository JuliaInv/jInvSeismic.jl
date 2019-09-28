using DelimitedFiles
using jInvVis
using PyPlot


m = DelimitedFiles.readdlm("ORIG_FWI_FC1_GN5.dat");

imshow(m'); colorbar();
