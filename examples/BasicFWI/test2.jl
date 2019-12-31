using SymEngine
using KrylovMethods

m = 4;
p = 2;
n = 3;

q = 2;

Z1 = [2 1 ;1 2; 1 1]
Z2 = [1 2 3; 1 1 1]
Z = Z1*Z2;
Z1s = [symbols("Z1_$i$j") for i in 1:m, j in 1:p];
Z2s = [symbols("Z2_$i$j") for i in 1:p, j in 1:n];
R = [symbols("R_$i") for i in 1:q];
Mul = [symbols("M_$i$j") for i in 1:m*n, j in 1:q];
G = Mul * transpose(Mul);
A = Vector{Basic}(undef, m*p);
Zs = reshape(Z1s * Z2s, (m*n, 1));
# R  = reshape(R, (m*q, 1));
# rhs = zeros(m *p);
# for mj = 1:p
#     v = zeros(m*n);
#     for j =1:n
#         v[m*(j-1) + 1] = Z2[mj, j];
#     end
#     for i = 1:m
#         A[((mj - 1) * m + i)] = dot(v, Zs);
#         rhs[((mj - 1) * m + i)] = dot(v, Z);
#         a = copy(v);
#         circshift!(v, a, 1);
#         # println(a)
#         # w = a
#     end
# end
Zd = Matrix{Basic}(undef, m * p, m * n);
for i = 1:m*p
    for j = 1:m*n
        Zd[i,j] =Basic(0);
    end
end


    for j=1:p
    for i=1:m
    for k=1:n
            Zd[(j-1)* m + i, (k-1) * m + i] = Z2s[j, k]
    end
    end
end

lhs = Zd * G * Zs;
for i = 1:m*p
    lhs[i] = expand(lhs[i]);
    println(lhs[i]);
end
rhs = Zd * Mul * R;

# function setVals(x, Z1s)
#     Z1s = reshape(x, (m,p));
#     return eval(Zs);
# end
# d = Dict()
# d[Z1s[1,1]] = 1;
# map()
# println(subs(Zs[1],d))
# abc = setVals([1 1 1 1 1 1], Z1s);

# res = KrylovMethods.cg((x)-> setVals, rhs, tol=1e-12, maxIter=100, out=2)[1];
# res = reshape(res, (m, p));
