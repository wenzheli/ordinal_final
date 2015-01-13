function [tar_set, tarNNIdx] = find_targetNN(data, label, nTar, M)

D = size(data,1);
N = size(data,2);

tar_set = zeros(nTar*N, 2);
tarNNIdx = zeros(nTar, N);

dist = dist_metric(data, data, M);

if size(label,1) == 1
    label = label';
end

dist = dist + double( repmat( label,1,N) ~= repmat( label',N, 1) ) * 1e10;

[dum, idx] = sort(dist,2);

tarNNIdx = idx(:,2:nTar+1)';

temp = repmat( 1:N, nTar, 1);
tar_set(:,1) = temp(:);
tar_set(:,2) = tarNNIdx(:);