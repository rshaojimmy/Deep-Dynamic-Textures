function [ train_x, train_y, test_x, test_y, deve_x, deve_y ] = CNNsplitsamples_Dld( x, y, testidx, trainidx, develpidx, isBigTrain )
%SPLITBYSBJCT split the 3DMAD data into 7 5 5, following the setting of
%TIFS 14 paper
%   x(1-3).data

% onesubNum = 25;
onesubNum = 5;
% onesubNum = 1500;
% onesubNum = 1;

% if length(portion) ~= length(x)-1
%     error('portion cannot match subset number');
% end

%% generate splitIndex for each portion

test_sbjIdx = testidx;


train_sbjIdx= trainidx;
develp_sbjIdx= develpidx;


%%


%% split data: following the setting in the TIFS14 paper: 7+5 for training, 5 for testing

if isBigTrain == 1 % training set could be the sum of TRAIN & DEVELOP set
    TIFS14_train_sbjIdx = sort([train_sbjIdx' develp_sbjIdx']');
else
    TIFS14_train_sbjIdx = sort(train_sbjIdx);
end

TIFS14_dvlp_sbjIdx = sort(develp_sbjIdx);


TIFS14_test_sbjIdx = sort(test_sbjIdx);


for i_train = 1:length(TIFS14_train_sbjIdx)
    start_idx = (TIFS14_train_sbjIdx(i_train)-1)*onesubNum;
    TIFS14_train_Idx(i_train,:) = start_idx+1 : start_idx+onesubNum;
end

for i_test = 1:length(TIFS14_test_sbjIdx)
    start_idx = (TIFS14_test_sbjIdx(i_test)-1)*onesubNum;
    TIFS14_test_Idx(i_test,:) = start_idx+1 : start_idx+onesubNum;
end

for i_dvlp = 1:length(TIFS14_dvlp_sbjIdx)
    start_idx = (TIFS14_dvlp_sbjIdx(i_dvlp)-1)*onesubNum;
    TIFS14_dvlp_Idx(i_dvlp,:) = start_idx+1 : start_idx+onesubNum;
end

train_x = [];
train_y = [];

test_x = [];
test_y = [];

deve_x = [];
deve_y= [];

% for i_sub = 1:length(x)
%     train_x = [train_x; x(i_sub).data(TIFS14_train_Idx(:),:)];
%     train_y = [train_y; y(i_sub).lable(TIFS14_train_Idx(:),:)];
%     
%     test_x = [test_x; x(i_sub).data(TIFS14_test_Idx(:),:)];
%     test_y = [test_y; y(i_sub).lable(TIFS14_test_Idx(:),:)];
%     
%     deve_x = [deve_x; x(i_sub).data(TIFS14_dvlp_Idx(:),:)];
%     deve_y = [deve_y; y(i_sub).lable(TIFS14_dvlp_Idx(:),:)];
%     
% end

for i_sub = 1:length(x)
    train_x = [train_x; x(i_sub).data(TIFS14_train_Idx(:),:)];
    train_y = [train_y; y(i_sub).label(TIFS14_train_Idx(:),:)];
    
    test_x = [test_x; x(i_sub).data(TIFS14_test_Idx(:),:)];
    test_y = [test_y; y(i_sub).label(TIFS14_test_Idx(:),:)];
    
    deve_x = [deve_x; x(i_sub).data(TIFS14_dvlp_Idx(:),:)];
    deve_y = [deve_y; y(i_sub).label(TIFS14_dvlp_Idx(:),:)];
    
end

%

end

