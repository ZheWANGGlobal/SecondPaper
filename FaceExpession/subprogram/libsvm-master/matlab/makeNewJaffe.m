load Jaffe;

%% make new Y with three categories (tag : 8 9 10)
% JaffeY_Three_Categories = zeros(213,1);
% for i=1:213
%    switch(Y(i,:))
%        case 1
%            JaffeY_Three_Categories(i,:) = 8;
%        case 2
%            JaffeY_Three_Categories(i,:) = 8;
%        case 3
%            JaffeY_Three_Categories(i,:) = 9;
%        case 4
%            JaffeY_Three_Categories(i,:) = 10;
%        case 5
%            JaffeY_Three_Categories(i,:) = 9;
%        case 6
%            JaffeY_Three_Categories(i,:) = 10;
%        case 7
%            JaffeY_Three_Categories(i,:) = 11;       
%    end
%    save JaffeY_Three_Categories.mat JaffeY_Three_Categories;
% end
%% make six X and six Y
% ind1 = find(Y==1);
% X_happy = X(ind1,:);
% save X_happy.mat X_happy;
% 
% ind2 = find(Y==2);
% X_sad = X(ind2,:);
% save X_sad.mat X_sad;
% 
% ind3 = find(Y==3);
% X_surprise = X(ind3,:);
% save X_surprise.mat X_surprise;
% 
% ind4 = find(Y==4);
% X_anger = X(ind4,:);
% save X_anger.mat X_anger;
% 
% ind5 = find(Y==5);
% X_disgust = X(ind5,:);
% save X_disgust.mat X_disgust;
% 
% ind6 = find(Y==6);
% X_fear = X(ind6,:);
% save X_fear.mat X_fear;
% 
% ind7 = find(Y==7);
% X_neutral = X(ind7,:);
% save X_neutral.mat X_neutral;
%% make three categories X
% JaffeX_happySad8 = zeros(31+31,4096);
% JaffeX_happySad8(1:31,1:4096) = X_happy(:,:);
% JaffeX_happySad8(32:62,1:4096) = X_sad(:,:);
% 
% JaffeX_SurDis9 = zeros(30+29,4096);
% JaffeX_SurDis9(1:30,:) = X_surprise;
% JaffeX_SurDis9(31:59,:) = X_disgust;
% 
% JaffeX_AngerFear10 = zeros(30+32,4096);
% JaffeX_AngerFear10(1:30,:) = X_anger;
% JaffeX_AngerFear10(31:62,:) = X_fear;
% 
% save JaffeX_happySad8.mat JaffeX_happySad8;
% save JaffeX_SurDis9.mat JaffeX_SurDis9;
% save JaffeX_AngerFear10.mat JaffeX_AngerFear10;
%% make matrix X with all three categories and make correponding Y
JaffeX_Six = zeros(62+59+62,4096);
JaffeX_Six(1:62,:) = JaffeX_happySad8(:,:);
JaffeX_Six(63:62+59,:) = JaffeX_SurDis9(:,:);
JaffeX_Six(62+60:62+59+62,:) = JaffeX_AngerFear10(:,:);

JaffeY_ThreeCate = zeros(62+59+62,1);
JaffeY_ThreeCate(1:62,:) = 8;
JaffeY_ThreeCate(63:62+59,:)=9;
JaffeY_ThreeCate(62+60:62+59+62,:) = 10;

JaffeY_Six = zeros(31+31+30+29+30+32,1);
JaffeY_Six(1:31,:) = 1;
JaffeY_Six(32:62,:) = 2;
JaffeY_Six(63:92,:) = 3;
JaffeY_Six(93:121,:) = 5;
JaffeY_Six(122:151,:) = 4;
JaffeY_Six(152:183,:) = 6;

save JaffeX_Six.mat JaffeX_Six;
save JaffeY_ThreeCate.mat JaffeY_ThreeCate;
save JaffeY_Six.mat JaffeY_Six;
