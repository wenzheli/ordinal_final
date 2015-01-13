function [intervals_new,max_accuracy] = update_interval( intervals, L, data, y, va_data, va_y )
% fix metric L, and update intervals, currently doing exaustive search
% for each interval, we divide it into K picecs, and try all the possible
% combinations and return the best one
% arguments:
%   intervals:      old intervals,  intervals{i}:  ith interval, which 
%                      is row vector. 
%   L:              matrix learne
%   data:           N * D data matrix
%   y:              data label, whic is colume vector. 
%   intervals_new:  new intervals to return


K = 10;

% for each possible interval, caculate the knn accuracy, return the 
% best interval gives highest accuracy for validation data set. 
max_iter = 4;
max_column = 1;
max_accuracy = 0;
N = size(data,1);
num_column = size(data, 2);
prev_accu = 0;
 
for iter = 1: max_iter
    % iterate all the columnes, and get the best one.
    disp(iter);
    flag = 1;
    for i = 1: num_column
        disp(i);
        curr_interval = intervals{i};   % row vector
        % divide current_interval into K pieces, and search
        % for the best end points 
        all_permutations = permute_intervals(curr_interval, K);
        for j = 1: size(all_permutations)
            fprintf('j = %d\n', j);
            % add candidate into the intervals. 
            intervals{i} = all_permutations(j,:);
            [accu, accu0] = compute_accuracy2(intervals, L, data, y, va_data, va_y);
            
            res(j) = accu;
            % save the current best
            if accu > max_accuracy
                fprintf('new accu %g\n', accu);
                max_accuracy = accu;
                max_column = i;
                tmp_interval = all_permutations(j,:);
                flag = 0;
            end
        end
        % restore back 
        intervals{i} = curr_interval;  
    end
    
    if flag == 1
        break;
    end
    
    % replace the best candidate
    intervals_new = intervals;
    intervals_new{max_column} = tmp_interval;
end











