using DelimitedFiles
using jInvVis
using PyPlot


m = DelimitedFiles.readdlm("ES_FWI_FC5_5_GN1.dat");

imshow(m'); colorbar();
