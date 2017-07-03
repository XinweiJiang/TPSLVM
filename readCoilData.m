function data = readCoilData( path )
%READCOILDATA Summary of this function goes here
%   Detailed explanation goes here

nObj = 50;
nPose = 100;
dirpath = [path '\'];

data = {};

for i = 1:nObj
    num = 0;
    A = [];
    for j = 1:nPose
        filepath = [dirpath 'obj' num2str(i) '__' num2str(j) '.png'];
        if exist(filepath,'file') ~= 2
            continue;
        end
        B = imread(filepath);
        A = [A ; B(:)'];
        num = num+1;
    end
    if ~isempty(A)
        data{i} = A;
    end
end


end

