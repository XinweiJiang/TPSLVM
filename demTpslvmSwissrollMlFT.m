clc;clear all; close all; st = fclose('all');
dataSetName = 'SwissrollFT';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?


% Load data
for i = [50,100]
    [x, y] = loadSwissroll(i);
%     x = sgpNormalize( x );
    [ x, y ] = smgpSort( x, y );
    xx = x(1:3,:);
    yy = y(1:3,:);
    %     xx = x;yy = y;
    %     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 30, 30 );

    % Set up and optimise model
    latentDim = 2;
    [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end
