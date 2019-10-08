clc;clear;

addpath(genpath('../'));

testround = 1;
subjctNum = 1;
downSpRate = 2;
% portion = [3 4]; %supplement dataset
rank = 256;

output_dir =  '/home/comp/ruishao/Documents/face_antispoofing/testpic/motion/intra/sup/labelold/ptnew/converge/';

if ~isdir(output_dir)
    mkdir(output_dir);
end

saveDataDir = [output_dir, 'convgdat3'];


load('../../../../feat_result/sup/motion/conv3_3/x_of_all.mat');
load('../../../../feat_result/sup/motion/conv3_3/y_old.mat');

load('../../../../motion/chldis/supnew.mat');


lable_iter_all = [];
scores_iter_all = [];

W0_iter_all = [];
Wi_iter_all = [];
D_iter_all = [];

param.alpha = 0.1;
param.nata = 0.1;
param.beta = 0.1;
param.theta = 0.1;

for j_iter = 1 : testround
    for i_iter = 1:subjctNum 
        exp_idx = (j_iter-1)*subjctNum + i_iter;
        testidx = test_all{exp_idx};
        trainidx = train_all{exp_idx};
        develpidx = develp_all{exp_idx};
        
        disp(['testround :', num2str(j_iter),' ', 'subjctNum :', num2str(i_iter)]);
        
        [ train_x, train_y, test_x, test_y, deve_x, deve_y ] = CNNsplitsamples_Dld( x, y, testidx, trainidx, develpidx,0);
            
        [allosevet, Dlosevet, W0lossvet, Wilossvet, Dvet, W0vet, Wivet] = mtmultk_convg( train_x, train_y, param );
                           
    end
end

save(saveDataDir,'allosevet', 'Dlosevet', 'W0lossvet', 'Wilossvet', 'Dvet', 'W0vet', 'Wivet', 'param');





