clc;clear all; close all; st = fclose('all');

Opt.iters = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'OilFT';
nTrOfEachClass = 20;
nTeOfEachClass = 1000;

% load data
% [x, y] = lvmLoadData('oil');
% x = sgpNormalize( x );
% [ x, y ] = smgpSort( x, y );
% xx = x(6,:);
% yy = y(6,:);

[x, y] = lvmLoadData('oil');
[ x, y ] = smgpSort( x, y );
[ x, y, xx, yy ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
if size(y,2) == 1
    nClass = length(unique(y));
else
    nClass = size(y,2);
end

for latentDim = [2]
    if nClass > 2
        [z, zz, retAcc]  = MultiSlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
%         [zretAcc]  = MultiSlltpslvm1VsRest( dataSetName, x, y, xx, yy, latentDim, Opt );
    else
        [z, zz, retAcc]  = BinarySlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
    end
end
