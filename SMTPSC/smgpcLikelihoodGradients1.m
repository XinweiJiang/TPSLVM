function [ nl, dnl ] = smgpcLikelihoodGradients1( hyper_w, model )
%MULTILAPLACEGP Summary of this function goes here
%   Detailed explanation goes here


nl = mpot(hyper_w);
dnl = mgrad1(hyper_w);

end

