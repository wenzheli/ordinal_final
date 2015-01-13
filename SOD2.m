function res = SOD2(x0, x, a, b, varM)

res = SOD(x, a, b);

nn = length(a);
D = size(varM,2);

V = x0(:,[a,b]);  % take all involved x0
s = zeros(1,D);
for j = 1:D
    p = V(j,:);
    q = varM(:,j);
    s(j) = sum( q(p) );
end

res = res + diag(s);
