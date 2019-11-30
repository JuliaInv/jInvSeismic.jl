using DelimitedFiles
using jInvVis
using PyPlot

figure();
m = DelimitedFiles.readdlm("ES_FWI_FC5_5_GN1.dat");
# # m = DelimitedFiles.readdlm("output_te/FWI_FC2_GN4.dat");
#
# ns = DelimitedFiles.readdlm("Ns1_GN1_FC1");
# maxN = maximum(ns[:])
# ind = findfirst(a->a==maxN, ns)
# ns[ind] =0.0
# ab = zeros(2278)
# xs = [5,1];

# for i=1:2278
#     ind = [((i-1) % 67) + 1, floor((i-1)/67) + 1];
#     # println(ind);
#     ab[i] = norm(xs - ind, 1)
#     # println(i, " ", m[i]);
# end
# ab= ab.+1;
# ns = ns./ab;
# m = reshape(ns, (67,34))
imshow(m'); colorbar();


# m = reshape(m, (67,34))
# m = m .* 100;
# println(m[1,5])
#
# imshow(m'); colorbar();
