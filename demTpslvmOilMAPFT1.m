clc;clear all; close all; st = fclose('all');
dataSetName = 'OilFT';
Opt.itersTrain = -500;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?
Opt.doTest = 0;
Opt.isBACK = 2;


for i = [100]
    % Load data
    load('Oil.mat');
    [ x, y ] = smgpSort( x, y );
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, i, 1000 );
%     xx = x(1:3,:);
%     yy = y(1:3,:);

    % Set up and optimise model
    for latentDim = [1,2,3]
        [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    end
end