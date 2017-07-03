clc;clear all; close all; st = fclose('all');
dataSetName = 'Usps5And3';
Opt.itersTrain = -500;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?
Opt.isBACK = 2;

% for i = [10:10:200]
for i = [100]
    % Load data
    [x, y, xx, yy] = loadBinaryUSPS(3,5);
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, i, 1000 );

    % Set up and optimise model
    for latentDim = [2]
        z = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    end
end