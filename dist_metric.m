function dist = dist_metric(X, Y, M)

% X: d*n
% Y: d*m
% M: d*d

% dist: n*m

n = size(X,2);
m = size(Y,2);

dist = repmat( diag(X'*M*X), 1, m) - 2 * X'*M*Y + repmat( diag(Y'*M*Y)', n, 1 );
