function [f, dL, n_cons] = lossfun_LMNN_end_point(L, data_orig, data, varM, lambda, tar_set, imp_set)

N = size(data,2);

dL = zeros( size(L) );

M = L'*L;
dist = dist_metric(data, data, M);   % pairwise distance
dist = dist(:);

lambda = size(imp_set,1) / size(tar_set,1);

%target neighbor distances
tarIndex = (tar_set(:,2)-1)*N + tar_set(:,1);
f = sum( dist( tarIndex ) ) / size(tar_set,1);

dL = dL + 2*L*SOD2(data_orig, data, tar_set(:,1), tar_set(:,2), varM) / size(tar_set,1);

% handeling imposters
tarDistIndex = (imp_set(:,2)-1)*N + imp_set(:,1);
impDistIndex = (imp_set(:,3)-1)*N + imp_set(:,1);

dist_diff = dist(tarDistIndex) - dist(impDistIndex) + 1;
pos_idx = find(dist_diff > 0);

n_cons = length(pos_idx);

f = f + lambda * sum( dist_diff(pos_idx) ) / size(imp_set,1);

dL = dL + lambda * 2 * L *( SOD2(data_orig, data, imp_set(pos_idx,1), imp_set(pos_idx,2), varM) - SOD2(data_orig, data, imp_set(pos_idx,1), imp_set(pos_idx,3), varM) ) / size(imp_set,1);