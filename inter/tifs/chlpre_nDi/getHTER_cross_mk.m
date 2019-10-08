function [HTER_test, FFR_star, FLR_star ] = getHTER_cross_mk( scores_test, test_x, test_y)
%GETHTER_ITER Summary of this function goes here
%   Detailed explanation goes here



%[tpr,fpr,thresholds] = roc(deve_y,scores_Linear_deve(:,2))

[FLR, TLR, T] = perfcurve(test_y, scores_test, 1);
FFR = 1-TLR;
[valu,starIdx] = min(abs(FLR - FFR));

tau_star = T(starIdx(1));
pridict_test_tau = scores_test>tau_star;

Pnum = sum(test_y == 1);
Nnum = sum(test_y ~= 1);
FFR_star = sum((pridict_test_tau==0)&(test_y == 1))/Pnum;
FLR_star = sum((pridict_test_tau==1)&(test_y == 0))/Nnum;
HTER_test = (FFR_star+FLR_star)/2;

end



