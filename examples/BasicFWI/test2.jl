using JLD
using Statistics
using KrylovMethods
using jInvVis
using PyPlot
# function calculateZ2(misfitCalc::Function, p::Integer, nsrc::Integer,
# 	nrec::Integer, nwork::Integer,
# 	numOfCurrentProblems::Integer, Wd::Array, HinvPs::Array,
# 	pMisCurrent::Array{MisfitParam}, currentSrcInd::Array, Z1::Matrix, alpha::Float64)
#
# 	println("misfit at start:: ", misfitCalc())
# 	rhs = zeros(ComplexF64, (p, nsrc));
# 	mSizeVec = size(Z1, 1);
#
# 	lhs = zeros(ComplexF64, (p,p));
#
# 	for i = 1:nwork:numOfCurrentProblems
# 		mergedSources = zeros(ComplexF64, (mSizeVec, nsrc))
# 		mergedDobs = zeros(ComplexF64, (nrec, nsrc))
# 		mergedWd = zeros(ComplexF64, (nrec, nsrc))
# 		for l=0:(nwork-1)
# 			mergedSources[:, currentSrcInd[i+l]] = pMisCurrent[i+l].pFor.Sources
# 			mergedDobs[:, currentSrcInd[i+l]] = pMisCurrent[i+l].dobs[:,:,1]
# 			mergedWd[:, currentSrcInd[i+l]] = Wd[i+l]
# 		end
# 		pm = pMisCurrent[i]
# 		lhs += (abs(mean(mergedWd))^2) .* Z1' * HinvPs[i] * HinvPs[i]' * Z1;
# 		rhs += (abs(mean(mergedWd))^2) .* Z1' * HinvPs[i] * (-HinvPs[i]' * mergedSources + mergedDobs);
# 	end
#
# 	lhs += alpha * I;
#
# 	return lhs\rhs;
# end
p=10
nsrc=15
alpha1 = 3e-1
alpha2 = 3e-1
betaS = 1e1
m=34
n=67


Z1 = load("zs.jld", "z1")
Z1abs = zeros(size(Z1,1), 1)
for i = 1:size(Z1,1)
	Z1abs[i] = norm(Z1[i,:])

end
Z2 = load("zs.jld", "z2")
NS = Z1 * Z2;
x = abs.(NS)
minimum(x)
maximum(x)
median(x)
a = x[x .< 1e-3]

figure();
imshow(reshape(Z1abs, (331, 116))')
# imshow(reshape(x, (580,331))')
colorbar();
# wd = load("ext.jld", "wd")
# hps = load("ext.jld", "hps")
# src = load("ext.jld", "src")

# dobs = load("ext2.jld", "dobs")
# wd = load("ext2.jld", "wd")
# hps = load("ext2.jld", "hps")
# src = load("ext2.jld", "src")
#
# # wdA = (abs(mean(wd[1]))^2)
# # lhs = (abs(mean(wd[1]))^2) .* Z1' * hps[1] *  hps[1]' * Z1 + alpha2*I
# # rhs =  (abs(mean(wd[1]))^2) .* Z1' * hps[1] * (dobs[1] - hps[1]' * src[1])
# # function misfitCalc2()
# # 	sum = 0;
# # 	res = hps[1]' * (src[1] + Z1 * Z2) - dobs[1]
# # 	sum +=  dot(wdA .* res, wdA.*res);
# #
# # 	sum	+= alpha1 * norm(Z1)^2 + alpha2 * norm(Z2)^2;
# # 	return sum;
# # end
# #
#
# Z1 = rand(ComplexF64,(m*n, p)) .+ 0.01;
# Z2 = rand(ComplexF64, (p, nsrc)) .+ 0.01;
# function a(Z1,Z2)
# 	wdA = (abs(mean(wd))^2)
# 	lhs = (wdA) .* Z1' * hps[1] *  hps[1]' * Z1 + alpha2*I
# 	rhs =  (wdA) .* Z1' * hps[1] * (dobs - hps[1]' * src)
# 	function misfitCalc2()
# 		sum = 0;
# 		res = hps[1]' * (src + Z1 * Z2) - dobs
# 		sum += wdA * dot(res, res);
#
# 		sum	+= alpha1 * norm(Z1)^2 + alpha2 * norm(Z2)^2;
# 		return sum;
# 	end
#
# 	Z2old = copy(Z2)
# 	Z1old = copy(Z1)
# 	Z2 = zeros(ComplexF64, (p, nsrc));
# 	Z1 = zeros(ComplexF64, (m*n, p));
# 	println("Zero Z2: ", misfitCalc2())
# 	Z2 = Z2old
# 	Z1 = Z1old
#
#
#
# 	println("Misfit B4: ", misfitCalc2())
# 	Z2t = lhs\rhs
#
#
#
# 	println("Misfit at Z2: ", misfitCalc2())
#
# 	function MultOp(HPinv, R, Z2t)
# 		return HPinv' * R * Z2t
# 	end
#
# 	function MultOpT(HPinv, R, Z2t)
# 		return HPinv * R * Z2t'
# 	end
#
# 	function MultAll(avgWds, HPinvs, R, Z2t, alpha, stepReg)
# 		sum = zeros(ComplexF64, size(R))
# 		# for i = 1:length(avgWds)
# 			sum += MultOpT(HPinvs, (wdA) .* MultOp(HPinvs, R, Z2t), Z2t)
# 		# end
# 		return sum + alpha * R + stepReg * R
# 	end
#
# 	Rc = wdA .* (dobs - hps[1]' * src)
#
# 	rhs = zeros(ComplexF64, size(Z1))
# 	rhs = MultOpT(hps[1], Rc, Z2t) +  betaS * Z1
#
# 	Z1t = KrylovMethods.blockBiCGSTB(x-> MultAll(wdA, hps[1], x, Z2t, alpha1, betaS), rhs,x=Z1, out=2)[1];
# 	println("Misfit at Z1: ", misfitCalc2())
#
# 	return Z1t,Z2t
# end
# function start(Z1,Z2)
# 	Z1t = Z1
# 	Z2t = Z2
# 	for j = 1:10
# 			Z1t,Z2t = a(Z1t,Z2t)
# 		# println(Z1)
# 		# Z1_1, Z2_1 = a(Z1, Z2)
# 		# Z1_2, Z2_2= a(Z1_1, Z2_1)
# 		# Z1_3, Z2_3 = a(Z1_2, Z2_2)
# 		# Z1_4, Z2_4 = a(Z1_3, Z2_3)
# 		# Z1_5, Z2_5 = a(Z1_4, Z2_4)
#
# 	end
# end
# start(Z1,Z2)
