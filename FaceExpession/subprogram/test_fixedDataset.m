clear
load x_trn
load x_tst
load y_tst
load y_trn
load y_three    % 训练集的三大类别标签
load y_three_tst   % 测试集的三大类别标签
N1=20;

%归一化
x_trn = double(x_trn)./255;
x_tst = double(x_tst)./255;

global p1 knn
 knn=1;
p1=2^-8;
%% using KNN directly
[out]=cknear(knn,x_trn,y_trn,x_tst); 
Acc(1,1)=mean(out==y_tst);
%% using SVM directly
model = svmtrain(y_trn,x_trn,'-s 0 -t 2 -g 0.001 -c 100');
[out, accu, de] = svmpredict(y_tst, x_tst, model);
Acc(1,2)=mean(out==y_tst);

confusionMatrix_former = zeros(7,7);

for i=1:63
        confusionMatrix_former(out(i),y_tst(i))=confusionMatrix_former(out(i),y_tst(i))+1;
end

%% 4 5作为一个大类  1 3作为一个大类   2 6作为一个大类
% 建立新的大类标签
y_trnNewCate = zeros(120,1);
y_tstNewCate = zeros(63,1);
for i=1:120
   if(y_trn(i,:)==1||y_trn(i,:)==3)
       y_trnNewCate(i,:) = 8;
   end
   if(y_trn(i,:)==4||y_trn(i,:)==5)
       y_trnNewCate(i,:) = 9;
   end
   if(y_trn(i,:)==2||y_trn(i,:)==6)
       y_trnNewCate(i,:) = 10;
   end
end

for i=1:63
   if(y_tst(i,:)==1||y_tst(i,:)==3)
       y_tstNewCate(i,:) = 8;
   end
   if(y_tst(i,:)==4||y_tst(i,:)==5)
       y_tstNewCate(i,:) = 9;
   end
   if(y_tst(i,:)==2||y_tst(i,:)==6)
       y_tstNewCate(i,:) = 10;
   end
end

% 第一层训练与识别
model = svmtrain(y_trnNewCate,x_trn,'-s 0 -t 2 -g 0.01 -c 100');
[out_NewCate, accu, de] = svmpredict(y_tstNewCate, x_tst, model);

% 第二层
ind_trn8 = find(y_trn==1|y_trn==3); 
ind_trn9 = find(y_trn==4|y_trn==5);
ind_trn10 = find(y_trn==2|y_trn==6);
model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8,:),'-s 0 -t 2 -g 0.01 -c 100');
model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9,:),'-s 0 -t 2 -g 0.01 -c 100');
model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10,:),'-s 0 -t 2 -g 0.01 -c 100');
% 找出第一层预测结果，准备分别投入相应分类器进行第二层预测
ind8 = find(out_NewCate==8);
ind9 = find(out_NewCate == 9);
ind10 = find(out_NewCate == 10);

% 第二层预测
[out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8,:), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
[out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9,:), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
[out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10,:), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));
Acc(1,3)=  ( sum(out10==y_tst(ind10))+ sum(out9==y_tst(ind9)) + sum(out8==y_tst(ind8))   )/63;


 %% SVM new senario with two levels （未降维）
% % 训练第二层的三个分类器   第一大类记为类别8，第二大类记为类别9，第三大类记为类别10
% ind_trn8 = find(y_trn==1|y_trn==2); 
% ind_trn9 = find(y_trn==3|y_trn==5);
% ind_trn10 = find(y_trn==4|y_trn==6);
% model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8,:),'-s 0 -t 2 -g 0.01 -c 100');
% model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9,:),'-s 0 -t 2 -g 0.01 -c 100');
% model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10,:),'-s 0 -t 2 -g 0.01 -c 100');
% 
% 
% 
% % 第一层分类器训练及预测
% model = svmtrain(y_three,x_trn,'-s 0 -t 2 -g 0.01 -c 100');
% [out_three, accu, de] = svmpredict(y_three_tst, x_tst, model);
% 
% %找出第一层预测结果，准备分别投入相应分类器进行第二层预测
% ind8 = find(out_three==8);
% ind9 = find(out_three == 9);
% ind10 = find(out_three == 10);
% 
% % 第二层预测y_tst(ind8)
% [out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8,:), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
% [out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9,:), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
% [out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10,:), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));

