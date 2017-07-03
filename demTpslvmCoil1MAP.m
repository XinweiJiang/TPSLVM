clc;clear all; close all; st = fclose('all');
dataSetName = 'Coil1';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?


% for i = [10:10:200]
% for i = [2,3]
% Load data
load('Coil20.mat');
x = double(data{2})./255;
y = ones(size(x,1),1);
%     [ x, y ] = smgpSort( x, y );
%     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 60, 500 );
xx = x(1:3,:);
yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end
% end