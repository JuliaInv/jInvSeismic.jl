using DelimitedFiles
using jInvVis
using PyPlot

figure();
m = DelimitedFiles.readdlm("output/FWI_FC1_GN5.dat");

imshow(m'); colorbar();
