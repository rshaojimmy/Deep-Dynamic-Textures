function [ train_x, train_y, test_x, test_y] = splitKFold_cross_Dld( x_train, y_train, x_test,y_test,trainidx, testidx )
%SPLITBYSBJCT split the 3DMAD data into 7 5 5, following the setting of
%TIFS 14 paper
%   x(1-3).data





onesubjNum = 5;

%% generate splitIndex for each portion

train_sbjIdx= trainidx;

test_sbjIdx = testidx;

%%


%% split data: following the setting in the TIFS14 paper: 7+5 for training, 5 for testing
TIFS14_train_sbjIdx = train_sbjIdx;



TIFS14_test_sbjIdx = sort(test_sbjIdx);


for i_train = 1:length(TIFS14_train_sbjIdx)
    start_idx = (TIFS14_train_sbjIdx(i_train)-1)*onesubjNum;
    train_Idx(i_train,:) = start_idx+1 : start_idx+onesubjNum;
end

for i_test = 1:length(TIFS14_test_sbjIdx)
    start_idx = (TIFS14_test_sbjIdx(i_test)-1)*onesubjNum;
    test_Idx(i_test,:) = start_idx+1 : start_idx+onesubjNum;
end



train_x = [];
train_y = [];

test_x = [];
test_y = [];


for i_sub = 1:length(x_train)
    train_x = [train_x; x_train(i_sub).data(train_Idx(:),:)];
    train_y = [train_y; y_train(i_sub).label(train_Idx(:),:)];
    
    test_x = [test_x; x_test(i_sub).data(test_Idx(:),:)];
    test_y = [test_y; y_test(i_sub).label(test_Idx(:),:)];

    
end

%

end

