clc;clear all; close all; st = fclose('all');

Opt.iters = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
% Opt.isPlot = 1;
dataSetName = 'Usps5And3DC';
% dataLabels = [0,1,2,3,4];
nTrOfEachClass = 50;
nTeOfEachClass = 1000;
filename = ['retSlltpslvm' dataSetName 'Accuracy' ];
Acc = [];

% load data
% [x, y, xx, yy] = loadBinaryUSPS(3,5);
% [ x, y ] = smgpSort( x, y );
% [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
[x, y, xx, yy] = loadBinaryUSPS(3,5);
xx = [x(51:717,:); xx];
yy = [y(51:717,:); yy];
x1 = x(1:50,:);x2 = x(718:767,:);
x=[x1;x2];
y1 = y(1:50,:);y2 = y(718:767,:);
y=[y1;y2];
clear x1 x2 y1 y2;

if size(y,2) == 1
    nClass = length(unique(y));
else
    nClass = size(y,2);
end

% for latentDim = [1,2,3,4,5,7,9,10,13,15]
for latentDim = [2]
    if nClass > 2
        [z, zz, retAcc]  = MultiSlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
    else
        [z, zz, retAcc]  = BinarySlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
    end
        
    Acc = [Acc; [latentDim size(x,1) size(xx,1) retAcc]];
end

% Acc = sortrows(Acc);
save(filename, 'Acc');
