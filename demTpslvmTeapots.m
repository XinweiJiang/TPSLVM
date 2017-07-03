clc;clear all; close all; st = fclose('all');
dataSetName = 'TeapotsFT';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isMAP = 0;          % use MAP/ML?
Opt.isBACK = 1;         % use back constrains?      0-without BC; 1-BC with kbr ; 2-BC with mlp
Opt.isDYN = 3;          % use Dynamic?      0-without dyn;1-gpReversible;2-gpTime;3-gp

for b = [1]
    for d = [3]
        Opt.isBACK = b;         % use back constrains?
        Opt.isDYN = d;          % use Dynamic?
        % Load data
        load('teapots100.mat');
        x = double(teapots')./255;
        y = ones(size(x,1),1);
        xx = x(1:3,:);yy = y(1:3,:);

            % Set up and optimise model
        for latentDim = [2]
            [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
        end
    end
end