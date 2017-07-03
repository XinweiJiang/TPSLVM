function [retImg, retLabel, remain] = delCloseSample( img, label, howclose )
%DELCLOSESAMPLE Summary of this function goes here
%   Detailed explanation goes here

n2 = dist2(img, img);
n2 = n2+diag(1000*ones(size(n2,1),1));
[idxR,idxC] = find(n2<howclose);
n = length(idxR);
remain = [];

for i = 1:n
    l = idxR(i);r = idxC(i);
    if idxR(i) > idxC(i)
        l = idxC(i);r = idxR(i);
    end
    if ~ismember([l r], remain, 'rows')
        remain = [remain; [l r]];
    end
end

uR = unique(remain(:,2));
[N, D] = size(img);
nRep = length(uR);
retImg = zeros(N-nRep,D);retLabel = zeros(N-nRep,size(label,2));
counter = 0;
for i = 1:N
    if ismember(i, remain(:,2))
        continue;
    end
    counter = counter+1;
    retImg(counter,:) = img(i,:);
    retLabel(counter,:) = label(i,:);
end

end

