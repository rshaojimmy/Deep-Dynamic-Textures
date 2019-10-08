% This script take the input structured feature and returns the results of
% SVM-RBF without feature aggregation

% addpath(genpath('D:\research\codes\Face antispoofing\rPPG_code\chopVideo_MAD\classifier_Msvm'));


%% svm with linear kernel (MATLAB ORIGINAL SVM)
clear;
clc;
close all
%% parameter setting
vidPerS = 5;
issave = 1;

rank = 256;
addpath(genpath('..\.'));
addpath(genpath('C:\research\codes\libsvm-3.21\'));
output_dir = 'D:\research2\gdfile\journal\TIFS\experiment\results\inter\tifsdiffattck\tifs2\';


if ~isdir(output_dir)
    mkdir(output_dir);
end
param.nata = 0.1;
param.beta = 0.1;
param.theta = 0.1;

%% load data


load('C:\research\codes\CNN_face_antispoofing\feat_results\3DMAD\CNN\vggnet\interlayers\motion\conv3_3\x_of_all.mat');
load('C:\research\codes\CNN_face_antispoofing\feat_results\3DMAD\CNN\vggnet\interlayers\motion\conv3_3\y_old.mat');


y_ours = y;
x_ours = x;

load('D:\research2\datasets_info\face_antispoofing\idiap\motion_fix2\all\x.mat');
load('D:\research2\datasets_info\face_antispoofing\idiap\motion_fix2\all\y.mat');

y_others = y;
x_others = x;


%% load train data
%load('E:\data\faceAnti\3DMAD\featureForClassify\subClass\label\y.mat');

crossProtcl = 'others2ours';

if strcmp(crossProtcl, 'ours2others') %train on ours, test on 3DMAD
    
    train_x{1} = [x_ours(1).data; x_ours(2).data; x_ours(3).data];
    train_y{1} = [y_ours(1).label; y_ours(2).label; y_ours(3).label];
    
    train_x{2} = [x_ours(1).data; x_ours(2).data; x_ours(3).data];
    train_y{2} = [y_ours(1).label; y_ours(2).label; y_ours(3).label];
    
    
    test_x{1} = cell2mat(x_others.train(:));
    test_y{1} = y_others.train;
    
    test_x{2} = cell2mat(x_others.dvlp(:));
    test_y{2} = y_others.test;
    
else if strcmp(crossProtcl, 'others2ours') %train on 3DMAD, test on ours    
        
    test_x{1} = [x_ours(1).data; x_ours(2).data; x_ours(3).data];
    test_y{1} = [y_ours(1).label; y_ours(2).label; y_ours(3).label];
    
    test_x{2} = [x_ours(1).data; x_ours(2).data; x_ours(3).data];
    test_y{2} = [y_ours(1).label; y_ours(2).label; y_ours(3).label];
    
    
    train_x{1} = cell2mat(x_others.train(:));
    train_y{1} = y_others.train;
    
    train_x{2} = cell2mat(x_others.test(:));
    train_y{2} = y_others.test;
    
    end
end


%% load test data

roundNum = 2;
lable_iter = [];
scores_iter = [];

for i_round = 1:roundNum

        
    disp(['testround:', num2str(i_round)]);
    
    new_train_x = train_x{i_round};
    new_train_y = train_y{i_round};
    
    new_test_x = test_x{i_round};
    new_test_y = test_y{i_round};

    [ new_train_x ] = norm2( new_train_x );
    [ new_test_x ] = norm2( new_test_x );
    
    linear_SVMModel = libsvmtrain(train_y,new_train_x, '-t 0');

    [ HTER_test(counter) ] = getHTER_iter_lib( linear_SVMModel, new_test_x, test_y);
    
    disp(['HTER: ',num2str(HTER_test(i_round))]);

    
%     lable_iter(:,i_round) = [new_test_y];
%     scores_iter(:,i_round) = [scores_test];
    lable_iter = [lable_iter, new_test_y'];
    scores_iter = [scores_iter, scores_test'];
    
    W0_iter{i_round} = W0;
    Wi_iter{i_round} = Wi;
    D_iter{i_round} = D;

    clear W0
    clear Wi
    clear D
   
end
meanHTER_test = mean(HTER_test);
stdHTER_test = std(HTER_test);

lable_sum = lable_iter;
score_sum = scores_iter;
% lable_sum = reshape(lable_iter, [1, size(lable_iter,1)*size(lable_iter,2)]);
% score_sum = reshape(scores_iter, [1, size(scores_iter,1)*size(scores_iter,2)]);

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
        'HTER_test', 'meanHTER_test', 'stdHTER_test',...
         'EER', 'W0_iter', 'Wi_iter', 'D_iter');
end

