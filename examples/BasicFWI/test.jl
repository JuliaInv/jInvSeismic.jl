# m=20;
# p=4;
# n=10;
#
# Z1 = 10 * rand(m,p);
# Z2 = 10 * rand(p,n);
#
# Ztot = Z1 * Z2;
#
# # Z1 = ones(m,n);
# a = zeros(m*n);
# println(typeof(a));
# println(a[1:3]);
# g = zeros(m * p, m*p);
# v = zeros(m *p, m*n);
# for i=1:m
# 	for j=1:p
# 		# v = zeros(m * n);
# 		startIdx = (i - 1) * n + 1;
# 		endIdx = (i - 1) * n + n;
# 		v[(i-1) * p + j, startIdx:endIdx] = Z2[j, :];
#
# 		startIdx = (i - 1) * p + 1;
# 		endIdx = (i - 1) * p + p;
# 		for k = 1:m
# 			for l = 1:p
# 				g[(i-1) * p + j, (k-1)*p + l] = (Z2[j,:])' * (Z2[l,:]);
# 			end
# 		end
# 		# g[(i-1) * p + j ,startIdx:endIdx] = (Z2[j,:])' * Z2';
# 	end
# end
#
# lu(g);
#
# Zv = reshape(Ztot, (m*n, 1));
# rhs = v * Zv;
#
# Z1rv = g\rhs;
using KrylovMethods


function dZ1x(x,Z2, m, p)
	println("start Wx");
	Z1x = zeros(m* p);
	for mj = 1:p
		w = zeros(m*p);

		for j =1:p
			w[m*(j-1) + 1] = (Z2[mj,:])' *  Z2[j,:] ;
		end
		for i = 1:m
			# println(w')
			Z1x[((mj - 1) * m + i)] = dot(w, x);
			a = copy(w);
			circshift!(w, a, 1);
			# println(a)
			# w = a
		end
	end
	println("end Wx");
	return Z1x;
end


function calculateZ1(Z2, Z)
	A = zeros(m*p);
	Zv = reshape(Z, (m*n,1))

	for mj = 1:p
		v = zeros(m*n);
		for j =1:n
			v[m*(j-1) + 1] = Z2[mj, j];
		end
		for i = 1:m
			A[((mj - 1) * m + i)] = dot(v, Zv);
			a = copy(v);
			circshift!(v, a, 1);
			# println(a)
			# w = a
		end
	end

	res = KrylovMethods.cg((x)-> dZ1x(x, Z2, m ,p) , A, tol=1e-12, maxIter=100, out=2)[1];
	res = reshape(res, (m, p));
end

# function calculateZ2(Z1, Z)
# 	A = zeros(p * n);
# 	Zv = reshape(Z, (m*n,1))
#
# 	for mj = 1:p
# 		v = zeros(m*n);
# 		for j =1:n
# 			v[m*(j-1) + 1] = Z2[mj, j];
# 		end
# 		for i = 1:m
# 			A[((mj - 1) * m + i)] = dot(v, Zv);
# 			a = copy(v);
# 			circshift!(v, a, 1);
# 			# println(a)
# 			# w = a
# 		end
# 	end
#
# 	res = KrylovMethods.cg((x)-> dZ1x(x, Z2, m ,p) , A, tol=1e-12, maxIter=100, out=2)[1];
# 	res = reshape(res, (m, p));
# end

function dZ2x(x, Z1, p, n)
	println("start Wx");
	A = zeros(n*p, n*p);
	Z2x = zeros(p * n);
	for mj = 1:p
		w = zeros(p * n);

		for j = 1:p
			w[j] = (Z1[:,mj])' *  Z1[:,j] ;
		end

		for i = 1:n
			println(w')
			A[:, p *(i - 1) + mj] = w'
			Z2x[p *(i - 1) + mj] = dot(w,x);
			a = copy(w);
			circshift!(w, a, p);
		end


			# println(a)
			# w = a

	end
	println("end Wx");
	# return Z2x;
	return A;
end


m = 3;
p = 2;
n = 3;

Z1 = [2 1 ;1 2; 1 1]
Z2 = [1 2 3; 1 1 1]

# println("b4 rand");
# Z1 = 10 * rand(m, p);
# Z2 = 10 * rand(p , n);
# println("after rand");
Z = Z1 * Z2;


W = zeros(m*p, m*p);

W[1, 1] = (Z1[:,1])' *  Z1[:,1] ;
W[1, 2] = (Z1[:,1])' *  Z1[:,2] ;

W[2, 1] = (Z1[:,2])' *  Z1[:,1] ;
W[2, 2] = (Z1[:,2])' *  Z1[:,2] ;

W[3, 3] = (Z1[:,1])' *  Z1[:,1] ;
W[3, 4] = (Z1[:,1])' *  Z1[:,2] ;

W[4, 3] = (Z1[:,2])' *  Z1[:,1] ;
W[4, 4] = (Z1[:,2])' *  Z1[:,2] ;

W[5, 5] = (Z1[:,1])' *  Z1[:,1] ;
W[5, 6] = (Z1[:,1])' *  Z1[:,2] ;

W[6, 5] = (Z1[:,2])' *  Z1[:,1] ;
W[6, 6] = (Z1[:,2])' *  Z1[:,2] ;


# v[1, 1] =(Z2[1,:])' *  Z2[1,:] ;
# v[1, 4] =(Z2[1,:])' *  Z2[2,:] ;
#
# v[2, 2] =(Z2[1,:])' *  Z2[1,:] ;
# v[2, 5] =(Z2[1,:])' *  Z2[2,:] ;
#
# v[3, 3] =(Z2[1,:])' *  Z2[1,:] ;
# v[3, 6] =(Z2[1,:])' *  Z2[2,:] ;
#
# v[4, 1] =(Z2[2,:])' *  Z2[1,:] ;
# v[4, 4] =(Z2[2,:])' *  Z2[2,:] ;
#
# v[5, 2] =(Z2[2,:])' *  Z2[1,:] ;
# v[5, 5] =(Z2[2,:])' *  Z2[2,:] ;
#
# v[6, 3] =(Z2[2,:])' *  Z2[1,:] ;
# v[6, 6] =(Z2[2,:])' *  Z2[2,:] ;

# v = [1 0 0 2 0 0 3 0 0;
# 	0 1 0 0 2 0 0 3 0;
# 	0 0 1 0 0 2 0 0 3;
# 	1 0 0 1 0 0 1 0 0;
# 	0 1 0 0 1 0 0 1 0;
# 	0 0 1 0 0 1 0 0 1];

v = [2 1 1 0 0 0 0 0 0;
	1 2 1 0 0 0 0 0 0;
	0 0 0 2 1 1 0 0 0 ;
	0 0 0 1 2 1 0 0 0;
	0 0 0 0 0 0 2 1 1;
	0 0 0 0 0 0 1 2 1];

# #
# #
Zv = reshape(Z, (m*n,1))
A = v* Zv;

res = calculateZ1(Z2, Z);


C = dZ2x([1 1 1 1 1 1], Z1, p, n);
# println("starting cg");
# res = KrylovMethods.cg((x)-> Wx(x, Z1, Z2, m ,p) , A, tol=1e-12, maxIter=100, out=2)[1];


# res = reshape(res, (m, p));
res = reshape(W\A, (p, n));
# println(norm(res.-Z2))
