function imp_set = find_imposter_nImp(data, label, tarNNIdx, nImp)

N = length(label);
nTar = size(tarNNIdx,1);
imp_set = zeros(N*nImp*nTar,3);

if size(label,1) == 1
    label = label';
end

all_dist =  dist_metric(data, data, eye(size(data,1)));

cc = 0;
for i = 1:N
    dist = all_dist(:,i);
    dist( label == label(i) ) = 1e10;
    
    [dum, idx] = sort(dist);
    idx = idx(1:nImp);
    
    imp_set_i = zeros(nTar*nImp,3);
    imp_set_i(:,1) = i;
    
    tarMat = repmat(tarNNIdx(:,i)',nImp,1);
    imp_set_i(:,2) = tarMat(:);
    
    imp_set_i(:,3) = repmat(idx,nTar,1);
    
    imp_set(cc+1:cc+nImp*nTar,:) = imp_set_i;
    cc = cc + nImp*nTar;
end