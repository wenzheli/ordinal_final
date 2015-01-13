function aa = calc_profit(n_year)

incubator_share = 0.35;
fund_share = 0.3;
IRR = 0.5;

fund_1 = 5;  % million 
fund_2 = 30;
fund_3 = 60;

revenue_1 = fund_1;
for t = 1:n_year
    revenue_1 = revenue_1 + revenue_1 * IRR;
end
profit_1 = ( revenue_1 - fund_1 ) * 0.8 * fund_share * incubator_share;

revenue_2 = fund_2;
for t = 2:n_year
    revenue_2 = revenue_2 + revenue_2 * IRR;
end
profit_2 = ( revenue_2 - fund_2 ) * 0.8 * fund_share * incubator_share;

revenue_3 = fund_3;
for t = 4:n_year
    revenue_3 = revenue_3 + revenue_3 * IRR;
end
profit_3 = ( revenue_3 - fund_3 ) * 0.8 * fund_share * incubator_share;

aa = profit_1 + profit_2 + profit_3;