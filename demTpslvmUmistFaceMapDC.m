clc;clear all; close all; st = fclose('all');
dataSetName = 'UmistFaceDC';
Opt.itersTrain = -200;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?
% Opt.doTest = 1;

Acc = [];
nTr = 5;    %number of training data in each class

% Load data
load('umistface_112x92.mat');
xa = []; ya = []; x = []; y = []; xx= []; yy = [];
for i = 1:20
    xa = [double(datacell{i})./255];
    ya = [i*ones(size(datacell{i},1),1)];
    x = [x; xa(1:nTr,:)];
    y = [y; ya(1:nTr,:)];
    xx = [xx; xa(nTr+1:size(xa,1),:)];
    yy = [yy; ya(nTr+1:size(xa,1),:)];
end
% x = double([datacell{1};datacell{2};datacell{3};datacell{4};datacell{5};datacell{6};datacell{7};datacell{8};datacell{9};datacell{10}])./255;
% y = ones(size(x,1),1);   
% xx = x(1:3,:);
% yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]%,3,5,10,15,20,25,30,35,40,45,50]
    [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    Acc = [Acc; [latentDim size(y,1) size(yy,1) retAcc]];
end

filename = ['retTpslvm' dataSetName 'Accuracy' ];
save(filename, 'Acc');
% save(['Tpslvm' dataSetName 'forClassify'], 'x','y','xx','yy','z','zz');
