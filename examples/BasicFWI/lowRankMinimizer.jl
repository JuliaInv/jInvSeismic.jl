using KrylovMethods
using JLD
using jInv.LinearSolvers
using Statistics
m = 67;
p = 10;
n = 34;

q = 58;

Z1 = 10 * rand(ComplexF64,(m,p)); #Initial guess
Z2 = 10 * rand(ComplexF64,(p,n)); #Initial guess

alpha = 1e-2;

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

# calculate A*x where A*x = (∂Z/∂Z2) * Wd^2 * M' * M * Z1*x + α*x
# In our case - M = P'*H^-1, Z = Z1 * Z2
function coeffsZ2(x, Z1, M, WdSqr)
	Z = Z1 * reshape(x, (p,n));
	Z = reshape(Z, (m*n, 1));
	return calculatedZ2x((M' .* WdSqr') * (M * Z), Z1) + alpha * x;
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

# calculate A*x where A*x = (∂Z/∂Z1) * Wd^2 * M' * M * Z(x) + α*x
# In our case - M = P'*H^-1, Z = Z1 * Z2
function coeffsZ1(x, Z2, M, WdSqr)
	Z = reshape(x, (m,p)) * Z2;
	Z = reshape(Z, (m*n, 1));
	return calculatedZ1x((M' .* WdSqr') * (M * Z), Z2) + alpha * x;
end

#function for solving min_Z1_Z2 ||b - PHinv * (Z1*Z2)||_F_Wd + α * ||Z1||_F + α * ||Z2||_F
# Where b = Dobs - PHinv * Q
function minimize(b, PHinv, Z1 ,Z2, Wd)
	WdSqr = Wd .^ 2;
	t = 1e-5;
	# Misfit for single Source is ||b - PHinv*Z1*Z2)||_F_Wd + α * ||Z1||_F + α * ||Z2||_F
	misfitCalc() = norm((b - PHinv * reshape(Z1 * Z2, (m*n , 1))).*Wd) + alpha * norm(Z1) + alpha * norm(Z2);
	misfitNorm = 10;

	println("misfit at start: ", misfitCalc());
	while misfitNorm > t
		prevMisfit = misfitCalc();

		#rhs is Wd^2 * ∂Z/∂Z1 * P' * H^-1 * b where Z = Z1*Z2
		rhs = calculatedZ1x((PHinv' .* WdSqr') * b, Z2);

		#calculate x such that Ax = rhs, coeffsZ1 calculates Ay for a vector y
		res = KrylovMethods.cg((x)-> coeffsZ1(x, Z2, PHinv, WdSqr) , Vector(rhs[:]), tol=1e-8, maxIter=100,
		x=complex(reshape(Z1, (m*p,1))[:]), out=2)[1];

		Z1 = reshape(res, (m, p));

		println("Misfit after Z1 calc: ", misfitCalc());
		return Z1,Z2;
		#rhs is Wd^2 * ∂Z/∂Z2 * P' * H^-1 * b where Z = Z1*Z2
		rhs = calculatedZ2x((PHinv' .* WdSqr') * b, Z1);

		#calculate x such that Ax = rhs, coeffsZ2 calculates Ay for a vector y
		res = KrylovMethods.cg((x)-> coeffsZ2(x, Z1, PHinv, WdSqr) , Vector(rhs[:]), tol=1e-8,
		maxIter=100, x=complex(reshape(Z2, (p*n,1))[:]), out=2)[1];
		Z2 = reshape(res, (p, n));

		misfit = misfitCalc();
		println("Misfit after both Z1 and Z2: ", misfit);
		misfitNorm = abs(prevMisfit - misfit);
	end
	return Z1, Z2;
end

PH = load("SavedVals2.jld", "hinv")
DOBS = load("SavedVals2.jld", "dobs")
WD = load("SavedVals2.jld", "wd")
SRC = load("SavedVals2.jld", "q1")

# Misfit for single Source is ||Dobs - P'*(H^-1)*(Q + Z1*Z2)||_F_Wd + alpha * ||Z1||_F + alpha * ||Z2||_F
misfitCalc() = norm((DOBS - PH * SRC - PH * reshape(Z1 * Z2, (m*n , 1))) .* WD) + alpha * norm(Z1) + alpha * norm(Z2);

println("Misfit before minimization ", misfitCalc());

Z1, Z2 = minimize(DOBS - PH * SRC, PH, Z1, Z2, WD);
println(misfitCalc())
