clear
% load Jaffe
load x_trn
load x_tst
load y_tst
load y_trn
load y_three    % ѵ��������������ǩ
load y_three_tst   % ���Լ�����������ǩ
% clear Y1;
% Y=Y1;
N1=20;
% X=double(X)./255;
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

%% 4 5��Ϊһ������ѵ��
% �����µ�y_�����ǩ
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

% ��һ��ѵ����ʶ��
model = svmtrain(y_trnNewCate,x_trn,'-s 0 -t 2 -g 0.01 -c 100');
[out_NewCate, accu, de] = svmpredict(y_tstNewCate, x_tst, model);

% �ڶ���
ind_trn8 = find(y_trn==1|y_trn==3); 
ind_trn9 = find(y_trn==4|y_trn==5);
ind_trn10 = find(y_trn==2|y_trn==6);
model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8,:),'-s 0 -t 2 -g 0.01 -c 100');
model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9,:),'-s 0 -t 2 -g 0.01 -c 100');
model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10,:),'-s 0 -t 2 -g 0.01 -c 100');
% �ҳ���һ��Ԥ������׼���ֱ�Ͷ����Ӧ���������еڶ���Ԥ��
ind8 = find(out_NewCate==8);
ind9 = find(out_NewCate == 9);
ind10 = find(out_NewCate == 10);

% �ڶ���Ԥ��
[out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8,:), model8);
Acc(1,3)=mean(out8==y_tst(ind8));
[out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9,:), model9);
Acc(1,4)=mean(out9==y_tst(ind9));
[out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10,:), model10);
Acc(1,5)=mean(out10==y_tst(ind10));
Acc(1,6)=  ( sum(out10==y_tst(ind10))+ sum(out9==y_tst(ind9)) + sum(out8==y_tst(ind8))   )/63;




 %% SVM new senario with two levels ��δ��ά��
% % ѵ���ڶ��������������   ��һ�����Ϊ���8���ڶ������Ϊ���9�����������Ϊ���10
% ind_trn8 = find(y_trn==1|y_trn==2); 
% ind_trn9 = find(y_trn==3|y_trn==5);
% ind_trn10 = find(y_trn==4|y_trn==6);
% model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8,:),'-s 0 -t 2 -g 0.01 -c 100');
% model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9,:),'-s 0 -t 2 -g 0.01 -c 100');
% model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10,:),'-s 0 -t 2 -g 0.01 -c 100');
% 
% 
% 
% % ��һ�������ѵ����Ԥ��
% model = svmtrain(y_three,x_trn,'-s 0 -t 2 -g 0.01 -c 100');
% [out_three, accu, de] = svmpredict(y_three_tst, x_tst, model);
% 
% %�ҳ���һ��Ԥ������׼���ֱ�Ͷ����Ӧ���������еڶ���Ԥ��
% ind8 = find(out_three==8);
% ind9 = find(out_three == 9);
% ind10 = find(out_three == 10);
% 
% % �ڶ���Ԥ��y_tst(ind8)
% [out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8,:), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
% [out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9,:), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
% [out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10,:), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));

%% SVM new senario with two levels ��PCA��ά��
% % ����PCA��ѵ�����Ͳ��Լ����н�ά
% [eigvectorPCA,eigvaluePCA] = PCA2(double(x_trn),3974);
% x_trn_PCA = double(x_trn)*eigvectorPCA;
% x_tst_PCA = double(x_tst)*eigvectorPCA; 
% 
% % ѵ���ڶ��������������   ��һ�����Ϊ���8���ڶ������Ϊ���9�����������Ϊ���10
% ind_trn8 = find(y_trn==1|y_trn==2); 
% ind_trn9 = find(y_trn==3|y_trn==5);
% ind_trn10 = find(y_trn==4|y_trn==6);
% model8 = svmtrain(y_trn(ind_trn8),x_trn_PCA(ind_trn8),'-s 0 -t 2 -c 100');
% model9 = svmtrain(y_trn(ind_trn9),x_trn_PCA(ind_trn9),'-s 0 -t 2 -c 100');
% model10 = svmtrain(y_trn(ind_trn10),x_trn_PCA(ind_trn10),'-s 0 -t 2 -c 100');
% 
% % ��һ�������ѵ����Ԥ��
% model = svmtrain(y_three,x_trn_PCA,'-s 0 -t 0 -c 100');
% [out_three, accu, de] = svmpredict(y_three_tst, x_tst_PCA, model);
% 
% %�ҳ���һ��Ԥ������׼���ֱ�Ͷ����Ӧ���������еڶ���Ԥ��
% ind8 = find(out_three==8);
% ind9 = find(out_three == 9);
% ind10 = find(out_three == 10);
% 
% 
% % �ڶ���Ԥ��
% [out8, accu, de] = svmpredict(y_tst(ind8), x_tst_PCA(ind8), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
% [out9, accu, de] = svmpredict(y_tst(ind9), x_tst_PCA(ind9), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
% [out10, accu, de] = svmpredict(y_tst(ind10), x_tst_PCA(ind10), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));

%% SVM new senario with two levels ��using LDA��
% %%% ��һ�������ѵ����Ԥ��
% % % ��һ�㽵ά
% % [V_LDA_three,inde] = LDA(x_trn, y_three,6);
% % x_trn_fistLevelDimensionReduction=x_trn*V_LDA_three;
% % x_tst_fistLevelDimensionReduction=x_tst*V_LDA_three;
% % ѵ����Ԥ��
% model = svmtrain(y_three,x_trn,'-s 0 -t 0 -c 100');
% [out_three, accu, de] = svmpredict(y_three_tst, x_tst, model);
% 
% %%% �ҳ���һ��Ԥ������׼���ֱ�Ͷ����Ӧ���������еڶ���Ԥ��
% ind8 = find(out_three==8);
% ind9 = find(out_three == 9);
% ind10 = find(out_three == 10);
% 
% %%% ѵ���ڶ��������������   ��һ�����Ϊ���8���ڶ������Ϊ���9�����������Ϊ���10
% ind_trn8 = find(y_trn==1|y_trn==2); 
% ind_trn9 = find(y_trn==3|y_trn==5);
% ind_trn10 = find(y_trn==4|y_trn==6);
% %���ηֱ�ά
% [V_LDA_8,inde] = LDA(x_trn(ind_trn8), y_trn(ind_trn8),1);
% x_trn_in8=x_trn(ind_trn8)*V_LDA_8;
% X_tst_in8=x_tst(ind8)*V_LDA_8;
% 
% [V_LDA_9,inde] = LDA(x_trn(ind_trn9), y_trn(ind_trn9),1);
% x_trn_in9=x_trn(ind_trn9)*V_LDA_9;
% X_tst_in9=x_tst(ind9)*V_LDA_9;
% 
% [V_LDA_10,inde] = LDA(x_trn(ind_trn10), y_trn(ind_trn10),1);
% x_trn_in10=x_trn(ind_trn10)*V_LDA_10;
% X_tst_in10=x_tst(ind10)*V_LDA_10;
% 
% % ������������ѵ��
% model8 = svmtrain(y_trn(ind_trn8),x_trn_in8,'-s 0 -t 2 -c 100');
% model9 = svmtrain(y_trn(ind_trn9),x_trn_in9,'-s 0 -t 2 -c 100');
% model10 = svmtrain(y_trn(ind_trn10),x_trn_in10,'-s 0 -t 2 -c 100');
% 
% %%% �ڶ���Ԥ��
% [out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
% [out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
% [out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));

%% SVM new senario with two levels ��using SMFA��
% %%% ��һ�������ѵ����Ԥ��
% % ��һ�㽵ά
% V_SMFA_three = sparse_MFA(x_trn, y_three,20);
% x_trn_fistLevelDimensionReduction=x_trn*V_SMFA_three;
% x_tst_fistLevelDimensionReduction=x_tst*V_SMFA_three;
% % ѵ����Ԥ��
% model = svmtrain(y_three,x_trn,'-s 0 -t 0 -c 100');
% [out_three, accu, de] = svmpredict(y_three_tst, x_tst, model);
% 
% %%% �ҳ���һ��Ԥ������׼���ֱ�Ͷ����Ӧ���������еڶ���Ԥ��
% ind8 = find(out_three==8);
% ind9 = find(out_three == 9);
% ind10 = find(out_three == 10);
% 
% %%% ѵ���ڶ��������������   ��һ�����Ϊ���8���ڶ������Ϊ���9�����������Ϊ���10
% ind_trn8 = find(y_trn==1|y_trn==2); 
% ind_trn9 = find(y_trn==3|y_trn==5);
% ind_trn10 = find(y_trn==4|y_trn==6);
% %���ηֱ�ά
% V_SMFA_8 = sparse_MFA(x_trn(ind_trn8), y_trn(ind_trn8),20);
% x_trn_in8=x_trn(ind_trn8)*V_SMFA_8;
% X_tst_in8=x_tst(ind8)*V_SMFA_8;
% 
% V_SMFA_9 = sparse_MFA(x_trn(ind_trn9), y_trn(ind_trn9),20);
% x_trn_in9=x_trn(ind_trn9)*V_SMFA_9;
% X_tst_in9=x_tst(ind9)*V_SMFA_9;
% 
% V_SMFA_10 = sparse_MFA(x_trn(ind_trn10), y_trn(ind_trn10),20);
% x_trn_in10=x_trn(ind_trn10)*V_SMFA_10;
% X_tst_in10=x_tst(ind10)*V_SMFA_10;
% 
% % ������������ѵ��
% model8 = svmtrain(y_trn(ind_trn8),x_trn_in8,'-s 0 -t 2 -c 100');
% model9 = svmtrain(y_trn(ind_trn9),x_trn_in9,'-s 0 -t 2 -c 100');
% model10 = svmtrain(y_trn(ind_trn10),x_trn_in10,'-s 0 -t 2 -c 100');
% 
% %%% �ڶ���Ԥ��
% [out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8), model8);
% Acc(1,3)=mean(out8==y_tst(ind8));
% [out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9), model9);
% Acc(1,4)=mean(out9==y_tst(ind9));
% [out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10), model10);
% Acc(1,5)=mean(out10==y_tst(ind10));
