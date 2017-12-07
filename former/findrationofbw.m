function [R]=findrationofbw(X,Y)
% Using this funtion to calculate the ration of bewteen-class scatter to
% the within-class scatter 
classnum=unique(Y);
Sb=0;
Sw=0;
m=mean(X);
 %[X]=normlizedata(X,'0-1');
for i=1:length(classnum)
    in=find(Y==classnum(i));
    X1=X(in,:);
    m1=mean(X1);
   dist=L2_distance(X1',m1');
   Sw=Sw+mean(dist);
   dist=L2_distance(m',m1');
   Sb=Sb+dist;
end
R=Sb/Sw;