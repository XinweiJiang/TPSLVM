clc;clear all; close all; st = fclose('all');

Opt.iters = -200;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'IrisFT';

% load data
[x, y] = loadData('iris.modified.data');
x = sgpNormalize( x );
[ x, y ] = smgpSort( x, y );
y = smgpTransformLabel( y );
xx = x(1:1,:);
yy = y(1:1,:);
if size(y,2) == 1
    nClass = length(unique(y));
else
    nClass = size(y,2);
end

% latentDim = 3;
for latentDim = [2]
    if nClass > 2
        [z, zz, retAcc]  = MultiSlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
    else
        [z, zz, retAcc]  = BinarySlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
    end
end
