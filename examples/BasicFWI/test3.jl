using KrylovMethods
using SymEngine


m = 600;
p = 10;
n = 300;

q = 50;

# Z1 = [2 1 ;1 2; 1 1; 3 4]
# Z2 = [1 2 3; 1 1 1]
Z1 = 10 * rand(m,p);
Z2 = 10 *rand(p,n);
println("=======1");
Z = Z1*Z2;
println("=======2");
Zv = reshape(Z, (m*n, 1));
# M = rand(q, m*n);
println("=======3");
M = rand(q, m*n);
# M = I;
Z1 = 10 * rand(m,p);
# println(Z1);
Z2 = 10 * rand(p,n);
println("=======4");
M[1:q, 1:q] += I;
M[1:q, 4:(3+q)] += I;

Q = M * Zv;

alpha = 1e-5;

function calculatedZ2x(x, Z1)
	A = zeros(p*n);
	for i=1:p
		v = zeros(m*n);
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

function coeffsZ2(x, Z1)
	Z = Z1 * reshape(x, (p,n));
	Z = reshape(Z, (m*n, 1));
	return calculatedZ2x(M' * (M * Z), Z1) + alpha * x;
end


function calculatedZ1(Z2)
	A = zeros(m*p, m*n);

	for mj = 1:p
		v = zeros(m*n);
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
	A = zeros(m*p);

	for mj = 1:p
		v = zeros(m*n);
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

function coeffsZ1(x, Z2)
	Z = reshape(x, (m,p)) * Z2;
	Z = reshape(Z, (m*n, 1));
	return calculatedZ1x(M' * (M * Z), Z2) + alpha * x;
end

function minimize(Dobs, PHinv,Z1 ,Z2)
	t = 1e-2;
	misfit = norm(Dobs - PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);
	misfitNorm = 10;
	println("misfit at start: ", misfit);
	while misfitNorm > t
		prevMisfit = misfit;
		rhs = calculatedZ1x(PHinv' * Dobs, Z2);
		res = KrylovMethods.cg((x)-> coeffsZ1(x, Z2) , Vector(rhs[:]), tol=1e-8, maxIter=100,  x=reshape(Z1, (m*p,1))[:], out=2)[1];
		Z1 = reshape(res, (m, p));
		misfit = norm(Dobs - PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);

		println(misfit);

		rhs = calculatedZ2x(PHinv' * Dobs, Z1);
		res = KrylovMethods.cg((x)-> coeffsZ2(x, Z1) , Vector(rhs[:]), tol=1e-8, maxIter=100, x=reshape(Z2, (p*n,1))[:], out=2)[1];
		Z2 = reshape(res, (p, n));

		misfit = norm(Dobs - PHinv * reshape(Z1 * Z2, (m*n , 1))) + alpha * norm(Z1) + alpha * norm(Z2);

		println(misfit);
		misfitNorm = abs(prevMisfit - misfit);
	end
	return Z1, Z2;
end


a1, a2 = minimize(Q,M, Z1, Z2);
println(norm(Q - M * reshape(a1 * a2, (m*n , 1))))
