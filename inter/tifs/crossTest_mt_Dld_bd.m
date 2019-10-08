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
addpath(genpath('../.'));

output_dir =  '/home/comp/ruishao/Documents/face_antispoofing/motion/results/intra/tifssupbd/';


if ~isdir(output_dir)
    mkdir(output_dir);
end
param.nata = 0.1;
param.beta = 0.1;
param.theta = 0.1;

%% load data


load('../../../feat_result/sup/motion/conv3_3/x_of_all.mat');
load('../../../feat_result/sup/motion/conv3_3/y_old.mat');


%% load train data
%load('E:\data\faceAnti\3DMAD\featureForClassify\subClass\label\y.mat');

crossProtcl = 'D3MAD2ours';

if strcmp(crossProtcl, 'ours2D3MAD') %train on ours, test on 3DMAD
    
    load('../../../motion/chldis/bd/ours2D3MAD_idx.mat');
    
    
else if strcmp(crossProtcl, 'D3MAD2ours') %train on 3DMAD, test on ours    
        
    load('../../../motion/chldis/bd/D3MAD2ours_idx.mat');
    end
end


%% load test data


isnorm = 'unnorm';

    
testidxnum = testidx;
trainidxnum = trainidx;


[ new_train_x, train_y, new_test_x, test_y] = splitKFold_cross_Dld( x, y, x,y,...
    trainidxnum, testidxnum );

[ W0, Wi, D ] = mtmultk( new_train_x, train_y, param, rank );
[ scores_test ] = test_multk( new_test_x, rank, W0, Wi, D );

[HTER_test ] = getHTER_cross_mk( scores_test, new_test_x, test_y);
disp(['HTER: ',num2str(HTER_test)]);


lable_iter= [test_y];
scores_iter = [scores_test];

W0_iter = W0;
Wi_iter= Wi;
D_iter = D;

   
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

    save(saveDataDir,'roc_x','roc_y','T','AUC',...
        'HTER_test', 'EER', 'W0_iter', 'Wi_iter', 'D_iter');
end

