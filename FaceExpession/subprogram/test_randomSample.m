clear
% load CK64Row.mat;
% load CKROW_Y.mat;
load Jaffe

% clear Y1;
% Y=Y1;
N1=20;
X=double(X)./255;
% CK64Row = uint8(CK64Row);
% CK64Row=double(CK64Row)./255;

global p1 knn
 knn=1;
p1=2^-8;
for loop=1:10
 [x_trn,y_trn,x_tst,y_tst,trainindex,testindex]=sample_random(X,Y,N1);
 %去除训练集中标签为7的样本
ind_excep7trn = find(y_trn~=7);
y_trn = y_trn(ind_excep7trn,:);
x_trn = x_trn(ind_excep7trn,:);
%去除测试集中标签为7的样本
ind_excep7tst = find(y_tst~=7);
y_tst = y_tst(ind_excep7tst,:);
x_tst = x_tst(ind_excep7tst,:);


 
 %% using KNN directly
[out]=cknear(knn,x_trn,y_trn,x_tst); 
Acc(loop,1)=mean(out==y_tst);
%% using SVM directly
model = svmtrain(y_trn,x_trn,'-s 0 -t 2 -g 0.001 -c 100');
[out, accu, de] = svmpredict(y_tst, x_tst, model);
Acc(loop,2)=mean(out==y_tst);

%计算直接SVM的结果混淆矩阵
confusionMatrix_former = zeros(7,7);
[row_tst,col_tst] = size(y_tst);
[row_trn,col_trn] = size(y_trn);
for i=1:row_tst
        confusionMatrix_former(out(i),y_tst(i))=confusionMatrix_former(out(i),y_tst(i))+1;
end

%% 大类8：1 3   大类9：4 5  大类10：2 6
% 建立新的y_大类标签
y_trnNewCate = zeros(row_trn,1);
y_tstNewCate = zeros(row_tst,1);
for i=1:row_trn
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

for i=1:row_tst
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
Acc(loop,3)=  ( sum(out10==y_tst(ind10))+ sum(out9==y_tst(ind9)) + sum(out8==y_tst(ind8))   )/row_tst;

%  tmp=1;

end
mean(Acc)
% Acc