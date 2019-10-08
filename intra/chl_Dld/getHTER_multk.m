function [ HTER_dvlp, HTER_test, FFR_star, FLR_star ] = getHTER_multk(  scores_deve, deve_y, scores_test, test_y)
%GETHTER_ITER Summary of this function goes here
%   Detailed explanation goes here


[FLR, TLR, T] = perfcurve(deve_y, scores_deve, 1);
%[tpr,fpr,thresholds] = roc(deve_y,scores_Linear_deve(:,2))

FFR = 1-TLR;
%starIdx = find(FLR == FFR);

[valu,starIdx] = min(abs(FLR - FFR));

tau_star = T(starIdx(1));

FFR_star = FFR(starIdx(1));
FLR_star = FLR(starIdx(1));
HTER_dvlp = (FFR_star+FLR_star)/2;


pridict_test_tau = scores_test>tau_star;

Pnum = sum(test_y == 1);
Nnum = sum(test_y ~= 1);
FFR_star = sum((pridict_test_tau==0)&(test_y == 1))/Pnum;
FLR_star = sum((pridict_test_tau==1)&(test_y == 0))/Nnum;
HTER_test = (FFR_star+FLR_star)/2;

end

