using KrylovMethods
using JLD
using jInv.LinearSolvers
using Statistics
m = 67;
p = 10;
n = 34;

q = 58;

# Z1 = [2 1 ;1 2; 1 1; 3 4]
# Z2 = [1 2 3; 1 1 1]
Z1 = 10 * rand(ComplexF64,(m,p));
Z2 = 10 *rand(ComplexF64,(p,n));
println("=======1");
Z = Z1*Z2;
println("=======2");
Zv = reshape(Z, (m*n, 1));
# M = rand(q, m*n);
println("=======3");
M = rand(ComplexF64, (q, m*n));
Z1 = 10 * rand(ComplexF64,(m,p));
# println(Z1);
Z2 = 10 * rand(ComplexF64,(p,n));
println("=======4");
M[1:q, 1:q] += I;
M[1:q, 4:(3+q)] += I;

Q = M * Zv;

alpha = 1e-7;

function calculatedZ2x(x, Z1)
	A = complex(zeros(p*n));
	for i=1:p
		v = complex(zeros(m*n));
		for j =1:m
			v[j] = Z1[j, i];
		end

		for j=1:n
			A[i + (j-1)*p] = dot(v, x);
			a = copy(v);
			circshift!(v, a, m);
		end
	end

	return A;
end

function coeffsZ2(x, Z1, M, WdSqr)
	Z = Z1 * reshape(x, (p,n));
	Z = reshape(Z, (m*n, 1));
	return calculatedZ2x((M' .* WdSqr') * (M * Z), Z1) + alpha * x;
end


function calculatedZ1(Z2)
	A = complex(zeros(m*p, m*n));

	for mj = 1:p
		v = complex(zeros(m*n));
		for j =1:n
			v[m*(j-1) + 1] = Z2[mj, j];
		end

		for i = 1:m
			A[((mj - 1) * m + i), :] = v;
			a = copy(v);
			circshift!(v, a, 1);
		end
	end
	return A;
end

function calculatedZ1x(x, Z2)
	A = complex(zeros(m*p));

	for mj = 1:p
		v = complex(zeros(m*n));
		for j =1:n
			v[m*(j-1) + 1] = Z2[mj, j];
		end

		for i = 1:m
			A[((mj - 1) * m + i)] = dot(v, x);
			a = copy(v);
			circshift!(v, a, 1);
		end
	end
	return A;
end

function coeffsZ1(x, Z2, M, WdSqr)
	Z = reshape(x, (m,p)) * Z2;
	Z = reshape(Z, (m*n, 1));
	return calculatedZ1x((M' .* WdSqr') * (M * Z), Z2) + alpha * x;
end

function minimize(Dobs, PHinv, Z1 ,Z2, Wd)
	WdSqr = 2 .* Wd .^ 2;
	t = 1e-5;
	misfitCalc() = norm(Wd.* Dobs - Wd.* PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);
	misfitNorm = 10;
	println("misfit at start: ", misfitCalc());
	while misfitNorm > t
		prevMisfit = misfitCalc();
		rhs = calculatedZ1x((PHinv' .* WdSqr') * Dobs, Z2);
		res = KrylovMethods.cg((x)-> complex(coeffsZ1(x, Z2, PHinv, WdSqr)) , Vector(complex(rhs[:])), tol=1e-8, maxIter=100,
		x=complex(reshape(Z1, (m*p,1))[:]), out=2)[1];
		Z1 = reshape(res, (m, p));
		# misfit = norm(WdSqr.*Dobs - WdSqr.*PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);

		println(misfitCalc());

		rhs = calculatedZ2x((PHinv' .* WdSqr') * Dobs, Z1);
		res = KrylovMethods.cg((x)-> complex(coeffsZ2(complex(x), Z1, PHinv, WdSqr)) , Vector(complex(rhs[:])), tol=1e-8,
		maxIter=100, x=complex(reshape(Z2, (p*n,1))[:]), out=2)[1];
		Z2 = reshape(res, (p, n));

		# misfit = norm(WdSqr.*Dobs - WdSqr.*PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);

		misfit = misfitCalc();
		println(misfit);
		misfitNorm = abs(prevMisfit - misfit);
	end
	return Z1, Z2;
end

PH = load("SavedVals.jld", "hinv")
DOBS = load("SavedVals.jld", "dobs")
WD = load("SavedVals.jld", "wd")
SRC = load("SavedVals.jld", "q1")
# PH = readdlm("PHINV_1", ',', ComplexF64)
# DOBS = readdlm("DOBS_1")
# WD = readdlm("WD_1")
# SRC = readdlm("Q1_1")
# * reshape(Z1 * Z2, (m*n , 1))
misfitCalc() = norm(WD.*DOBS - WD.*PH * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);

mis = misfitCalc();
#
# Wd = ones(size(Q))./(mean(abs.(Q)));
# WdSqr = Wd.^2;
WdSqr = WD.^2;
a1, a2 = minimize(Q,M, Z1, Z2, WdSqr);
# println(norm(WdSqr.*Q - WdSqr .* M * reshape(a1 * a2, (m*n , 1))) + alpha * norm(a1) + alpha * norm(a2))
