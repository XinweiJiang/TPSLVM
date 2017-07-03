clc;clear all; close all; st = fclose('all');
dataSetName = 'TeapotsFT';
Opt.itersTrain = -100;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?


% Load data
load('teapots100.mat');
x = double(teapots')./255;
y = ones(size(x,1),1);
%     [ x, y ] = smgpSort( x, y );
%     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 60, 500 );
xx = x(1:3,:);
yy = y(1:3,:);

    % Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end