function ordinal_metric(name_data, split)

% 1: nur   2: car    3: balance  4: syn

if name_data == 1
    load('nur_data.mat');
    
    [trX, trY, vaX, vaY, teX, teY] = get_split_data(all_X, all_Y, train_idx, valid_idx, test_idx, split);
    
    data = trX;
    y = trY;
    
    [val,idx] = sort(y);
    data = data(idx,:);
    y = y(idx);
    
    tst_data = teX;
    tst_y = teY;
    
    va_data = vaX;
    va_y = vaY;
    
elseif name_data == 2
    load('car_data.mat');
    
    [trX, trY, vaX, vaY, teX, teY] = get_split_data(all_X, all_Y, train_idx, valid_idx, test_idx, split);
    
    data = trX;
    y = trY;
    
    [val,idx] = sort(y);
    data = data(idx,:);
    y = y(idx);
    
    tst_data = teX;
    tst_y = teY;
    
    va_data = vaX;
    va_y = vaY;
    
    data(:,3) = data(:,3)-1;
    data(:,4) = data(:,4)/2;
    
    tst_data(:,3) = tst_data(:,3)-1;
    tst_data(:,4) = tst_data(:,4)/2;
    
    va_data(:,3) = va_data(:,3) - 1;
    va_data(:,4) = va_data(:,4)/2;
elseif name_data == 3
    load('bal_data.mat');
    
    [trX, trY, vaX, vaY, teX, teY] = get_split_data(all_X, all_Y, train_idx, valid_idx, test_idx, split);
    
    data = trX;
    y = trY;
    
    [val,idx] = sort(y);
    data = data(idx,:);
    y = y(idx);
    
    tst_data = teX;
    tst_y = teY;
    
    va_data = vaX;
    va_y = vaY;
end

% suppose all the attributes are ordinal
N = size(data,1);
D = size(data,2);
K = size(unique(y),1);

% normalize data
% for j = 1:D
%     max_val =  max( data(:,j) );
%     data(:,j) = data(:,j) / max_val;
%     va_data(:,j) = va_data(:,j) / max_val;
%     tst_data(:,j) = tst_data(:,j) / max_val;
% end

nK = 3;

% baseline - Euclidean
errEuc = knncl(eye(D), data', y', va_data', va_y', nK);
fprintf('Euclidean: training %f\t test %f\n', errEuc(1), errEuc(2));

% baseline - LMNN
[L, dum] = lmnn2(data', y', 3, 'maxiter',2000, 'quiet', 1);
errLMNN = knncl(L, data', y', va_data', va_y', nK);
fprintf('LMNN: training %f\t test %f\n', errLMNN(1), errLMNN(2));

% basline - Euclidean - binary
max_val = max(data);
bi_data = zeros(N, sum(max_val));
for i = 1:N
    sta_idx = 0;
    for j = 1:D
        bi_data(i, sta_idx+1:sta_idx+data(i,j)) = 1;
        sta_idx = sta_idx + max_val(j);
    end
end

bi_va_data = zeros( size(va_data,1), sum(max_val) );
for i = 1:size(va_data,1)
    sta_idx = 0;
    for j = 1:D
        bi_va_data(i, sta_idx+1:sta_idx+va_data(i,j)) = 1;
        sta_idx = sta_idx + max_val(j);
    end
end

errEuc = knncl(eye(size(bi_data,2)), bi_data', y', bi_va_data', va_y', nK);
fprintf('Euclidean-new: training %f\t test %f\n', errEuc(1), errEuc(2));

% baseline - LMNN - binary
[L, dum] = lmnn2(bi_data', y', 3, 'maxiter',2000, 'quiet', 1);
errLMNN = knncl(L, bi_data', y', bi_va_data', va_y', nK);
fprintf('LMNN-new: training %f\t test %f\n', errLMNN(1), errLMNN(2));

max_val = max(data);
bi_data = zeros(N, sum(max_val));
for i = 1:N
    sta_idx = 0;
    for j = 1:D
        bi_data(i, sta_idx+data(i,j)) = 1;
        sta_idx = sta_idx + max_val(j);
    end
end

bi_va_data = zeros( size(va_data,1), sum(max_val) );
for i = 1:size(va_data,1)
    sta_idx = 0;
    for j = 1:D
        bi_va_data(i, sta_idx+va_data(i,j)) = 1;
        sta_idx = sta_idx + max_val(j);
    end
end

errEuc_bi = knncl(eye(size(bi_data,2)), bi_data', y', bi_va_data', va_y', 1);

% baseline - LMNN - binary
[L, dum] = lmnn2(bi_data', y', 3, 'maxiter',1000, 'quiet', 1);
errLMNN_bi = knncl(L, bi_data', y', bi_va_data', va_y', nK);

% our method
MAX_ITER = 5;

% initialize end points
intervals = cell(D,1);
inter_len = zeros(D,1);
for i = 1:D
    % # of ordinal values for each dimension
    n = length(unique(data(:,i)));
    
    interval = linspace(1/(2*n), (2*n+1)/(2*n), n+1);
    intervals{i} = interval;
    inter_len(i) = length(interval);
end

% get target neighbors and imposters
nTar = 3;
nImp = 10;
[tar_set, tarNNIdx] = find_targetNN(data', y', nTar, eye(D));
imp_set = find_imposter_nImp(data', y', tarNNIdx, nImp);

lambda = 1;

L = eye(D);
for iter = 1:MAX_ITER
    % generate new data
    new_data = zeros(N,D);
    for i = 1:N
        for j = 1:D
            inter = intervals{j};
            new_data(i,j) = (inter( data(i,j) ) + inter( data(i,j)+1 ))/2;
        end
    end
    
    max_inter_len = max(inter_len);
    varM = zeros(max_inter_len-1, D);
    for j = 1:D
        inter = intervals{j};
        for i = 1:inter_len(j)-1
            varM(i,j) = ( inter(i+1)-inter(i) )^2 / 12;
        end
    end
    
    
    % fix end points, learn L
    L = learn_L_fix_end_point(L, data', new_data', varM, tar_set, imp_set, lambda);
    
    accu_L(iter) = compute_accuracy2(intervals, L, data, y, va_data, va_y);
 
    % fix L, search for end points
    % such that the validation accuracy is max
    %[intervals, accu_interval(iter)] = update_interval( intervals, L, data, y, va_data, va_y );
    [intervals, accu_interval(iter)] = update_interval_GD( intervals, L, data, y, va_data, va_y );
end

fname = ['result/Data_', num2str(name_data), '_S_', num2str(split), '_res.mat'];
save(fname, 'errEuc','errLMNN','errEuc_bi','errLMNN_bi','accu_L','accu_interval','intervals');