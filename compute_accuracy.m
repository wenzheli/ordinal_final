function accu = compute_accuracy(intervals, L, data, y0, va_data, va_y)
% compute the accuracy
total = size(va_data,1);
correct = 0;
D = size(data,2);  % dimensional of the feature vector
k = 3;    % knn

for i=1:size(va_data,1)
    disp(i);
    % for each data in validation set
    x = va_data(i,:);
    dists = zeros(size(data,1),1);   % colume vector
    for j=1:size(data,1)
        y = data(j,:);
        % construct matrix M to store all the expectations of distances
        % between x and y. Assume all the features are ordinal
        M = zeros(D,D);
        for d1=1:D
            x_d1_s = intervals{d1}(x(d1));
            x_d1_e = intervals{d1}(x(d1)+1);
            y_d1_s = intervals{d1}(y(d1));
            y_d1_e = intervals{d1}(y(d1)+1);
            for d2=1:D                
                if d1 == d2
                    % E[(x_d1-y_d1)^2]=E[x_d1^2]+E[y_d1^2]-2E[x_d1]E[y_d1] 
                    % E(x^2) = var(x) + E(x)^2
                    ext_diff = (x_d1_e-x_d1_s)^2/12 + (x_d1_e+x_d1_s)^2 /4 + (y_d1_e-y_d1_s)^2/12 + (y_d1_e+y_d1_s)^2 /4 - 2 *(x_d1_e+x_d1_s)*(y_d1_e+y_d1_s)/4;
                    
                else
                    % E[(x_d1-y_d1)(x_d2-y_d2)]=E[x_d1*x_d2]+E[y_d1*y_d2]-E[x_d1*y_d2]-E[x_d2*y_d1]
                    x_d2_s = intervals{d2}(x(d2));
                    x_d2_e = intervals{d2}(x(d2)+1);
                    y_d2_s = intervals{d2}(y(d2));
                    y_d2_e = intervals{d2}(y(d2)+1);
                    
                    ext_diff = (x_d1_s+x_d1_e)/2 * (x_d2_s+x_d2_e)/2 + (y_d1_s+y_d1_e)/2 * (y_d2_s+y_d2_e)/2 - (x_d1_s+x_d1_e)/2 * (y_d2_s+y_d2_e)/2 - (y_d1_s+y_d1_e)/2 *(x_d2_s+x_d2_e)/2; 
                end
                M(d1,d2)=ext_diff;
            end
        end
        
        dists(j) = trace(L * M);
    end
    
    % compute the best label using KNN
    [sortedValue, sortedIndexs] = sort(dists);  % row vector
    top_labels = y0(sortedIndexs(1:k));          
    best_label = mode(top_labels);
    
    if best_label == va_y(i)
        correct = correct + 1;
    end
end

accu = correct/total;

