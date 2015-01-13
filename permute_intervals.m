function all_permuations = permute_intervals(interval, K)
% return all the possible permutations of input interval. 
n = length(interval)-2;
d = (interval(end)-interval(1))/K;
interval_points = zeros(1, K-1);

interval_points(1)=interval(1)+d;
for i=2:K-1
    interval_points(i) = d + interval_points(i-1);
end

% compute all the permutations. choose n from K-1
combos = nchoosek(interval_points, n);
total = size(combos,1);
all_permutations = zeros(total, length(interval));
for i=1:total
    tmp = zeros(1,length(interval));
    tmp(2:end-1)=combos(i,:);
    tmp(1) = interval(1);
    tmp(end) = interval(end);
    all_permuations(i,:) = tmp;
end



