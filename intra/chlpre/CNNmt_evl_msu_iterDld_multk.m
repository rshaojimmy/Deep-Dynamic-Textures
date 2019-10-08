clc;clear;

addpath(genpath('../.'));

param.iternum = 80;
downSpRate = 2;
rank = 256;

output_dir =  '/home/comp/ruishao/Documents/face_antispoofing/motion/results/intra/tifs/msu/';

if ~isdir(output_dir)
    mkdir(output_dir);
end

saveFigDir = [output_dir, 'msuf_print_80'];
saveDataDir = [output_dir, 'msud_print_80'];


% load('../../../../feat_result/sup/motion/conv3_3/x_of_all.mat');
load('/home/comp/ruishao/Documents/face_antispoofing/feat_result/msu/print/x.mat');
load('/home/comp/ruishao/Documents/face_antispoofing/feat_result/msu/print/y.mat');


lable_iter_all = [];
scores_iter_all = [];

W0_iter_all = [];
Wi_iter_all = [];
D_iter_all = [];

param.alpha = 0.1;
param.nata = 0.1;
param.beta = 0.1;
param.theta = 0.1;


train_x = x.train;
test_x = x.test;

train_y = y.train;
test_y = y.test;


[ W0, Wi, D ] = mtmultk( train_x, train_y, param, rank, param.iternum);

[ scores_test ] = test_multk( test_x, rank, W0, Wi, D);


lable_iter= [test_y];
scores_iter = [scores_test];

lable_sum = reshape(lable_iter, [1, size(lable_iter,1)*size(lable_iter,2)]);
score_sum = reshape(scores_iter, [1, size(scores_iter,1)*size(scores_iter,2)]);

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
save(saveDataDir,'roc_x','roc_y','T','AUC','EER',...
    'W0', 'Wi', 'D','param');





