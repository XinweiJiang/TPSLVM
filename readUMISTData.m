function [datacell, data] = readUMISTData( path )
%READCOILDATA Summary of this function goes here
%   Detailed explanation goes here

dirpath = [path '\'];
NAME = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t'];
nObj = length(NAME);
nPose = 100;

datacell = {};
data = [];

for i = 1:nObj
    num = 0;
    A = [];
    for j = 0:nPose
        filepath = [dirpath '1' NAME(i) '\face\' '1' NAME(i) num2str(j,'%03d') '.pgm'];
        if exist(filepath,'file') ~= 2
            filepath = [dirpath '1' NAME(i) '\' '1' NAME(i) num2str(j,'%03d') '.pgm'];
            if exist(filepath,'file') ~= 2
                continue;
            end
        end
        B = imread(filepath);
        A = [A ; B(:)'];
        num = num+1;
    end
    if ~isempty(A)
        datacell{i} = A;
        data = [data; A];
    end
end


end

