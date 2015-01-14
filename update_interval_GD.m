function [new_intervals, accu] = update_interval_GD( intervals, L, data, y, tar_set, imp_set)
%    the function updates intervals given metric L using SGD (with
%    projection). The update proceeds as two-step iterate : 
%        1. update intervals using SGD
%        2. project learned intervals using QP programming with ordinal
%           constraints. 
%    
%       inter = intervals{i}:   $i$th interval, inter(1), and inter(end) are
%       the starting and end points, and we assume that:
%       -1 <= inter(1) <= inter(2) <= .... <= inter(end) <= 1
%       which is bounded by -1 and 1. 
%   returns 
%      new intervals,  accuracy. 


fprintf('Begin update intervals given L...\n');

%% parameters.......
MAX_ITER =  100;
step_size = 0.0001;
N = size(data, 1);
D = size(data, 2);
lamda = 1;
m = [];
inter_len = zeros(D,1);
start_idx = 1;
for i=1:length(intervals)    
    inter = intervals{i};
    inter_len(i) = length(inter);
    m(start_idx: start_idx + inter_len(i)-1) = inter;
    start_idx = start_idx + inter_len(i);
end

m = m';                                 

T = sum(inter_len);                     % total number of end points within intervals. 
A = cell(N,1);
C = cell(N,1);

G = L'*L;

%% pre-processing :  
%     prepare matrix A_i, C_i for each data(i,:) such that 
%     A_i*m = E(X_i)
%     C_i*m*m'*C_i' = Var(X_i)
for i=1:N
    tmp_A = zeros(D, T);
    tmp_C = zeros(D, T);
    for j=1:D
        inter = intervals{j};
        % fill out the matrix. 
        start_idx = sum(inter_len(1:j))-inter_len(j);
        tmp_A(j, start_idx + data(i,j): start_idx + data(i,j)+1) = 1/2;
        tmp_C(j, start_idx + data(i,j)) = -1;
        tmp_C(j, start_idx + data(i,j)+1) = 1;
    end
    A{i} = tmp_A;
    C{i} = tmp_C;
end

grad = zeros(T,T);
for itr=1: MAX_ITER
    itr
  
    % update intervals m using gradient descent 
    
    % for target sets
    for j=1:length(tar_set)
        s = tar_set(j,1);
        t = tar_set(j,2);
        grad = grad + 2 * (A{s}-A{t})'*G*(A{s}-A{t}) + 1/12* C{s}'*G*C{s} + 1/12*C{t}'*G*C{t};
    end
    
    % for imposters
    for j=1:length(imp_set)
        s = imp_set(j,1);
        t = imp_set(j,2);
        im = imp_set(j,3);
        tmp1 = (A{s}-A{t})'*G*(A{s}-A{t}) + 1/12* C{s}'*G*C{s} + 1/12*C{t}'*G*C{t};
        tmp2 = (A{s}-A{im})'*G*(A{s}-A{im}) + 1/12* C{s}'*G*C{s} + 1/12*C{im}'*G*C{im};
        if trace(m*m'*tmp1 - m*m'*tmp2 + 1) > 0
            grad = grad + 2 * lamda * (tmp1 - tmp2); 
        end
    end
    
    m = m - step_size * grad * m;
    m
    
    % project M by solving QP programming under ordinal constraints. 
    % min 1/2(m-\tilde(m))^2   s.t ordinal constraints. 
    %  min  m'Hm + f'm
    %  s.t  B*m <= b
    H = eye(T);
    B = zeros(T+D, T);
    b = zeros(T+D,1);
    
    % fill out the A and b
    row_idx = 1;
    col_idx = 1;
    for j=1:D
        for k=1:inter_len(j)
            B(row_idx + k-1, col_idx + k-1) = -1;
            B(row_idx + k, col_idx + k-1) = 1;
        end
        b(row_idx)=1;
        b(row_idx + inter_len(j))=1;
        row_idx = row_idx + inter_len(j) + 1;
        col_idx = col_idx + inter_len(j);
    end   
    m = quadprog(H,-m,B,b);
    m
end

% assign intervals.
start_idx = 1;
for i=1:D
    intervals{i} = m(start_idx : start_idx + inter_len(i)-1)';
    start_idx = start_idx + inter_len(i);
end












