function [] = plotZforTeapots( O )
%PLOTZFORTEAPOTS Summary of this function goes here
%   Detailed explanation goes here
BASEPATH = 'R:/';
SCALEY = 1.5;
SCALEX = SCALEY*101/76;
LWIDTH = 3;
nShow = 6;
nEachClass = 10;

h0 = figure; 
hAxes = axes('NextPlot','add');           %# Add subsequent plots to the axes,


baseX = 0;baseY = 10;
for i = 1:100
    nLevel = floor((i-1)/10)+1;
    nNum = mod(i-1,10)+1;
    
    d0=imrotate(reshape(O(i,:),76,101,3),180);
    image(d0,'Parent',hAxes,'XData',[baseX+(nNum-1)*SCALEX baseX+nNum*SCALEX],'YData',[baseY-nLevel*SCALEY baseY-(nLevel-1)*SCALEY],'CDataMapping','scaled');
end

axis image;
set(gca,'xtick',[]);
set(gca,'ytick',[]);
print(h0,'-depsc','-tiff', '-loose', '-r600', [BASEPATH 'dv_TeapotsOrigin'])


end

