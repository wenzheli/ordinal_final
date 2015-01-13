function [xTr, yTr, xVa, yVa, xTe, yTe] = generate_data_split(data, label, train_idx, valid_idx, test_idx, split) 

xTr = data( train_idx{split},:);
yTr = label( train_idx{split},:);

xVa = data( valid_idx{split},:);
yVa = label( valid_idx{split},:);

xTe = data( test_idx{split},:);
yTe = label( test_idx{split},:);
