clear
% load Jaffe
% load JaffeX_Six
% load JaffeY_Six
% load JaffeY_ThreeCate
load x_trn
load x_tst
load y_tst
load y_trn
load y_three
load y_three_tst
% clear Y1;
% Y=Y1;
N1=20;
% X=double(X)./255;

global p1 knn
 knn=1;
p1=2^-8;
C=100;
ker='linear';
ker='rbf';
% for loop=1:50
%  [x_trn,y_trn,x_tst,y_tst,trainindex,testindex]=sample_random(X,Y,N1);
% [x_trn,y_trn,x_tst,y_tst,trainindex,testindex]=sample_random(JaffeX_Six,JaffeY_Six,N1);
%  [~,P]=pca(x_trn,size(x_trn,1));
%  newx_trn=x_trn*P;
%  newx_tst=x_tst*P;
%   [out]=cknear(knn,newx_trn,y_trn,newx_tst); 
%   mean(out==y_tst)
 %% KNN
   [out]=cknear(knn,x_trn,y_trn,x_tst); 
%   Acc(loop,1)=mean(out==y_tst);
Acc(1,1)=mean(out==y_tst);
  %% SVM
%  [model,err,predictY,out]= SVC_lib_tt(x_trn,y_trn,ker,C,x_tst,y_tst);
%    Acc(loop,2)=1-err;
model = svmtrain(y_trn,x_trn,'-s 0 -t 0 -c 100');
[out, accu, de] = svmpredict(y_tst, x_tst, model);
% Acc(loop,2)=mean(out==y_tst);
Acc(1,2)=mean(out==y_tst);
%% SVM new senario
% % % predict three categories
% % y_trn_three = JaffeY_ThreeCate(trainindex,:);
% % model = svmtrain(y_trn_three,x_trn,'-s 0 -t 2 -c 100');
% % y_tst_three = JaffeY_ThreeCate(testindex,:);
% % [out, accu, de] = svmpredict(y_tst_three, x_tst, model);
% % ind8 = find(out==8);
% % ind9 = find(out == 9);
% % ind10 = find(out == 10);
% % 
% % %retrain 
% % ind_trn8 = find(y_trn==1.0|y_trn==2.0);
% % ind_trn9 = find(y_trn==3.0|y_trn==5.0);
% % ind_trn10 = find(y_trn==4.0|y_trn==6.0);
% % model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8),'-s 0 -t 2 -c 100');% model 8
% % model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9),'-s 0 -t 2 -c 100');
% % model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10),'-s 0 -t 2 -c 100');
% % 
% % [out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8), model8);
% % Acc(loop,3)=mean(out8==y_tst(ind8));
% % [out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9), model9);
% % Acc(loop,4)=mean(out9==y_tst(ind9));
% % [out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10), model10);
% % Acc(loop,5)=mean(out10==y_tst(ind10));

% save x_trn.mat x_trn
% save y_trn.mat y_trn
% save x_tst.mat x_tst
% save y_tst.mat y_tst

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind_trn8 = find(y_trn==1|y_trn==2); 
ind_trn9 = find(y_trn==3|y_trn==5);
ind_trn10 = find(y_trn==4|y_trn==6);
model8 = svmtrain(y_trn(ind_trn8),x_trn(ind_trn8),'-s 0 -t 2 -c 100');% model 8
model9 = svmtrain(y_trn(ind_trn9),x_trn(ind_trn9),'-s 0 -t 0 -c 100');
model10 = svmtrain(y_trn(ind_trn10),x_trn(ind_trn10),'-s 0 -t 0 -c 100');
% predict three categories
model = svmtrain(y_three,x_trn,'-s 0 -t 0 -c 100');
[out_three, accu, de] = svmpredict(y_three_tst, x_tst, model);
ind8 = find(out_three==8);
ind9 = find(out_three == 9);
ind10 = find(out_three == 10);

[out8, accu, de] = svmpredict(y_tst(ind8), x_tst(ind8), model8);
Acc(1,3)=mean(out8==y_tst(ind8));
[out9, accu, de] = svmpredict(y_tst(ind9), x_tst(ind9), model9);
Acc(1,4)=mean(out9==y_tst(ind9));
[out10, accu, de] = svmpredict(y_tst(ind10), x_tst(ind10), model10);
Acc(1,5)=mean(out10==y_tst(ind10));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% SMFA + SVM
   % sparse MFA without PCA
%      for i=1:20
%         V_SMFA_PCA=sparse_MFA(x_trn,y_trn,i); 
%         x_trn = double(x_trn);
%         x_tst= double(x_tst);
%         X_trn=x_trn*V_SMFA_PCA;
%         X_tst=x_tst*V_SMFA_PCA;
%         [out2]=cknear(knn,X_trn,y_trn,X_tst); 
%         res2(loop,i)=mean(out2==y_tst);
%      end
% % x_trn = double(x_trn);
% % x_tst= double(x_tst);
% % V_SMFA=sparse_MFA(x_trn,y_trn,10); 
% % X_trn=x_trn*V_SMFA;
% % X_tst=x_tst*V_SMFA;
% % model2 = svmtrain(y_trn,X_trn,'-s 0 -t 2 -c 100');
% % [out, accu, de] = svmpredict(y_tst, X_tst, model2);
% % Acc(loop,3)=mean(out==y_tst);
% end
% mean(Acc)
% Acc