using DelimitedFiles
using jInvVis
using PyPlot


m = DelimitedFiles.readdlm("TE_FWI_FC5_GN5.dat");

imshow(m'); colorbar();
