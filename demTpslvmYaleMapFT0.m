clc;clear all; close all; st = fclose('all');
dataSetName = 'YaleFT';
Opt.itersTrain = -200;
Opt.itersTest = -100;
Opt.nKnn = 5;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?

% Load data
load('Yale_64x64.mat');
x = fea/255;
y = gnd;
xx = x(1:5,:);
yy = y(1:5,:);

% Set up and optimise model
for latentDim = [1:2:20]
% for latentDim = [45]
    [z] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end

