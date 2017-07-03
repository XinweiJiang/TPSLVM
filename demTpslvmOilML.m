clc;clear all; close all; st = fclose('all');
dataSetName = 'OilNC';
Opt.itersTrain = -100;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?


% for i = [10:10:200]
for i = [2]
    % Load data
    load('Oil.mat');
    [ x, y ] = smgpSort( x, y );
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 60, 500 );
    xx = x(1:3,:);
    yy = y(1:3,:);
%     xx = x;yy = y;
%     [ x, y, xx, yy ] = spgDivTrainTestData( x, y, 30, 30 );
%     xx = x; yy = y;

    % Set up and optimise model
    latentDim = i;
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end