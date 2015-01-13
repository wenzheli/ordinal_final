function [accu, accu0] = compute_accuracy2(intervals, L, data, y, va_data, va_y)

% compute the accuracy

[N,D] = size(data);

new_data = zeros(N,D);
for i = 1:N
    for j = 1:D
        inter = intervals{j};
        new_data(i,j) = (inter( data(i,j) ) + inter( data(i,j)+1 ))/2;
    end
end

[M,D] = size(va_data);
new_va_data = zeros(M,D);
for i = 1:M
    for j = 1:D
        inter = intervals{j};
        new_va_data(i,j) = (inter( va_data(i,j) ) + inter( va_data(i,j)+1 ))/2;
    end
end

G = L'*L;
dist = dist_metric(new_va_data', new_data', G);

[val, idx] = min(dist,[],2);
pred = y(idx);

accu0 = sum( pred == va_y ) / length(va_y);

for j = 1:D
    inter_len(j) = length(intervals{j});
end

max_inter_len = max(inter_len);
varM = zeros(max_inter_len-1, D);
for j = 1:D
    inter = intervals{j};
    for i = 1:inter_len(j)-1
        varM(i,j) = ( inter(i+1)-inter(i) )^2 / 12;
    end
end

dist2 = zeros( size(dist) );
for i = 1:N
    tmp = 0;
    for j = 1:D
        tmp = tmp + G(j,j) * varM( data(i,j), j );
    end
    dist2(:,i) = tmp;
end

dist = dist + dist2;

[val, idx] = min(dist,[],2);
pred = mode( y([idx(:,1), idx(:,1), idx(:,1), idx(:,2), idx(:,2), idx(:,3), idx(:,3)]),2);

accu = sum( pred == va_y ) / length(va_y);