clc;clear all; close all; st = fclose('all');
dataSetName = 'RobotWireless';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
Opt.isMAP = 1;          % use MAP/ML?
% Opt.isBACK = 1;         % use back constrains?
% Opt.isDYN = 3;          % use Dynamic?

for b = [0,1,2]
    for d = [0,1,2,3]
        Opt.isBACK = b;         % use back constrains?
        Opt.isDYN = d;          % use Dynamic?
        
        % Load data
        load(dataSetName);
        xx = x(1:3,:);yy = y(1:3,:);

        % Set up and optimise model
        for latentDim = [2]
            [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
        end
    end
end