for t = 3:8
    aa(t) = calc_profit(t);
end

plot(1:8, aa, '*');