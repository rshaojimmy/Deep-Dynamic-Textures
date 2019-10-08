clc;clear;

addpath(genpath('../'));

testround = 2;
subjctNum = 8;
downSpRate = 2;
rank = 256;

output_dir =  '../../../../testpic/motion/intra/sup/labelold/2/';

if ~isdir(output_dir)
    mkdir(output_dir);
end

saveFigDir = [output_dir, 'fig'];
saveDataDir = [output_dir, 'data'];


load('../../../../feat_result/sup/motion/conv3_3/x_of_all.mat');
load('../../../../feat_result/sup/motion/conv3_3/y_old.mat');

load('../../../../motion/chldis/sup1D_e5.mat');


lable_iter_all = [];
scores_iter_all = [];

W0_iter_all = [];
Wi_iter_all = [];
D_iter_all = [];

param.nata = 0.1;
param.beta = 0.2;
param.theta = 0.1;

for j_iter = 1 : testround
    tic
    for i_iter = 1:subjctNum 
        exp_idx = (j_iter-1)*subjctNum + i_iter;
        testidx = test_all{exp_idx};
        trainidx = train_all{exp_idx};
        develpidx = develp_all{exp_idx};
        
        disp(['testround :', num2str(j_iter),' ', 'subjctNum :', num2str(i_iter)]);
        
        [ train_x, train_y, test_x, test_y, deve_x, deve_y ] = CNNsplitsamples_Dld( x, y, testidx, trainidx, develpidx,0);
        
        [ W0, Wi, D ] = mtmultk( train_x, train_y, param );
                
        [ scores_test ] = test_multk( test_x, rank, W0, Wi, D);
        [ scores_deve] = dvlp_multk( deve_x,  rank, W0, Wi, D); 
        

        [ HTER_dvlp(i_iter), HTER_test(i_iter) ] = getHTER_multk( scores_deve, deve_y, scores_test, test_y);
        disp(['HTER of dvlp/iter :', num2str(HTER_dvlp(i_iter)),' ', 'HTER of tst/iter :', num2str(HTER_test(i_iter))]);

        lable_iter(:,i_iter) = [deve_y; test_y];
        scores_iter(:,i_iter) = [scores_deve;scores_test];
        
        W0_iter{i_iter} = W0;
        Wi_iter{i_iter} = Wi;
        D_iter{i_iter} = D;
        
        clear W0
        clear Wi
        clear D

    end
    meanHTER_dvlp(j_iter) = mean(HTER_dvlp);
    meanHTER_test(j_iter) = mean(HTER_test);
    lable_iter_all  = [lable_iter_all , lable_iter];
    scores_iter_all = [scores_iter_all ,scores_iter];
    
    W0_iter_all = [W0_iter_all, W0_iter];
    Wi_iter_all = [Wi_iter_all, Wi_iter];
    D_iter_all = [D_iter_all, D_iter];
    
    toc
end

meanAllHTER_dvlp = mean(meanHTER_dvlp);
meanAllHTER_test = mean(meanHTER_test);
disp(['HTER of test:' num2str(meanAllHTER_test*100) , '%']);
disp(['HTER of develop:' num2str(meanAllHTER_dvlp*100) , '%']);

lable_sum = reshape(lable_iter_all, [1, size(lable_iter_all,1)*size(lable_iter_all,2)]);
score_sum = reshape(scores_iter_all, [1, size(scores_iter_all,1)*size(scores_iter_all,2)]);

lable_sum_down = downsample(lable_sum,downSpRate);
score_sum_down = downsample(score_sum,downSpRate);

[roc_x, roc_y, T,AUC,OPTROCPT] = perfcurve(lable_sum_down, score_sum_down, 1);

[val, EERindx] = min(abs(roc_x - (1-roc_y)));

EER = (roc_x(EERindx)+(1-roc_y(EERindx)))/2;

disp(['EER :' num2str(EER*100) , '%']);

h1 = plot(roc_x, 1-roc_y,'--xr',...
    'LineWidth',1,...
    'MarkerSize',4);

hold on;
xlim([-0.00,1.00]); ylim([-0.00,1.00]);
title(['Cropped Face ROC', ', AUC = ', num2str(AUC)]);
xlabel('False Living Rate');
ylabel('False Fake Rate');
grid on
grid minor
set(gca,'xtick',[0:0.2:1],'ytick',[0:0.2:1])

saveas(gcf,saveFigDir,'fig');
saveas(gcf,saveFigDir,'png');
save(saveDataDir,'roc_x','roc_y','T','AUC','EER','meanAllHTER_test','meanAllHTER_dvlp', 'W0_iter_all', ...
     'Wi_iter_all', 'D_iter_all','param');





