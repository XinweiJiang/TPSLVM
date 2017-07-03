clc;clear all; close all; st = fclose('all');
dataSetName = 'UmistFace1';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?


% Load data
load('umistface_112x92.mat');
x = []; y = [];
for i = 1:6
    x = [x; double(datacell{i})./255];
    y = [y; i*ones(size(datacell{i},1),1)];
end
% x = double([datacell{1};datacell{2};datacell{3};datacell{4};datacell{5};datacell{6};datacell{7};datacell{8};datacell{9};datacell{10}])./255;
% y = ones(size(x,1),1);   
xx = x(1:3,:);
yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end

