% This script take the input structured feature and returns the results of
% SVM-RBF without feature aggregation

% addpath(genpath('D:\research\codes\Face antispoofing\rPPG_code\chopVideo_MAD\classifier_Msvm'));


%% svm with linear kernel (MATLAB ORIGINAL SVM)
clear
clc
close all
%% parameter setting
vidPerS = 5;
issave = 1;

rank = 256;
addpath(genpath('../'));


output_dir =  '../../../testpic/motion/inter/nDi/';
if ~isdir(output_dir)
    mkdir(output_dir);
end
param.nata = 0.1;
param.beta = 0.1;
param.theta = 0.1;

%% load data

load('../../../feat_result/3DMAD/motion/conv3_3/x_of_all.mat');
load('../../../feat_result/3DMAD/motion/conv3_3/y_old.mat');
y_D3MAD = y;
x_D3MAD = x;


load('../../../feat_result/sup/motion/conv3_3/x_of_all.mat');
load('../../../feat_result/sup/motion/conv3_3/y_old.mat');
y_ours = y;
x_ours = x;


%% load train data
%load('E:\data\faceAnti\3DMAD\featureForClassify\subClass\label\y.mat');

crossProtcl = 'ours2D3MAD';

if strcmp(crossProtcl, 'ours2D3MAD') %train on ours, test on 3DMAD
    
    load('/home/comp/ruishao/Documents/face_antispoofing/testpic/motion/inter/256_itr30_new/ours2D3MAD_data.mat');
    load('/home/comp/ruishao/Documents/face_antispoofing/motion/chldis/ours2D3MAD_idx.mat');

    y_train = y_ours;
    x_train = x_ours;
    y_test = y_D3MAD;
    x_test = x_D3MAD;
%     train_subjctNum = 8;
%     test_subjctNum = 17;
%     portion = 6;
else if strcmp(crossProtcl, 'D3MAD2ours') %train on 3DMAD, test on ours    
        
        load('/home/comp/ruishao/Documents/face_antispoofing/testpic/motion/inter/256/D3MAD2ours_data.mat');
        load('/home/comp/ruishao/Documents/face_antispoofing/motion/chldis/D3MAD2ours_idx_old.mat');

        y_train = y_D3MAD;
        x_train = x_D3MAD;
        y_test = y_ours;
        x_test = x_ours;
%         train_subjctNum = 17;
%         test_subjctNum = 8;
%         portion = 12;
    end
end


%% load test data

roundNum = 20;

isnorm = 'unnorm';

counter = 0;
for i_round = 1:roundNum
    
    testidxnum = testidx{i_round};
    trainidxnum = trainidx{i_round};
    
    counter = counter+1;
    
    disp(['testround:', num2str(i_round)]);
    
    
    [ new_train_x, train_y, new_test_x, test_y] = splitKFold_cross_Dld( x_train, y_train, x_test,y_test,...
        trainidxnum, testidxnum );
    
    W0 = W0_iter{i_round};
    Wi = Wi_iter{i_round};
    D = D_iter{i_round};
    
    
%     [ scores_test ] = test_multk_nwi( new_test_x, rank, W0, D );
    [ scores_test ] = test_multk_nDi( new_test_x, W0, Wi );
    
    [HTER_test(i_round) ] = getHTER_cross_mk( scores_test, new_test_x, test_y);
    disp(['HTER: ',num2str(HTER_test(i_round))]);

    
    lable_iter(:,counter) = [test_y];
    scores_iter(:,counter) = [scores_test];
  
end
meanHTER_test = mean(HTER_test);

lable_sum = reshape(lable_iter, [1, size(lable_iter,1)*size(lable_iter,2)]);
score_sum = reshape(scores_iter, [1, size(scores_iter,1)*size(scores_iter,2)]);

[roc_x, roc_y, T,AUC,OPTROCPT] = perfcurve(lable_sum, score_sum, 1);

[val, EERindx] = min(abs(roc_x - (1-roc_y)));

EER = (roc_x(EERindx)+(1-roc_y(EERindx)))/2;


disp(['AUC ', num2str(AUC)]);
disp(['EER ', num2str(EER)]);


h1 = plot(1-roc_y,roc_x ,'--xr',...
    'LineWidth',2,...
    'MarkerSize',4)

hold on;
%legend('PCBwVA','Location','Best')

xlim([-0.00,1.00]); ylim([-0.00,1.00]);
title(['LBP cross-database ROC', ', AUC = ', num2str(AUC)]);
xlabel('False Fake Rate');
ylabel('False Living Rate');
grid on
grid minor
set(gca,'xtick',[0:0.2:1],'ytick',[0:0.2:1])

ffr_x = 1-roc_y;
flr_y = roc_x;

saveFigDir = [output_dir crossProtcl '_fig'];
saveDataDir = [output_dir, crossProtcl '_data'];

if(issave)
    saveas(gcf,saveFigDir,'fig');
    saveas(gcf,saveFigDir,'png');

    save(saveDataDir,'roc_x','roc_y','T','AUC','meanHTER_test', 'EER');
end

