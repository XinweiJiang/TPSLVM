clc;clear all; close all; st = fclose('all');
dataSetName = 'IsomapFace';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?


% Load data
load('isomap_face_data_sameremoved.mat');
% load('isomap_face_data.mat');
% x0 = images';
% y0 = ones(size(x0,1),1);
% 
% % Delete very close samples in the data set for numerical stability
% [x, y] = delCloseSample(x0, y0, 6);           

xx = x(1:3,:);
yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end