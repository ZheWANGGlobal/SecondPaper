clear
load Jaffe
load Jaffe32_row.mat
% clear Y1;
% Y=Y1;
N1=20;
X=double(X)./255;
Jaffe32_row = double(Jaffe32_row)./255;

global p1 knn
knn=1;
p1=2^-8;
C=100;
ker='linear';
ker='rbf';
for loop=1:3
[x_trn,y_trn,x_tst,y_tst,trainindex,testindex]=sample_random(Jaffe32_row,Y,N1);
%% sparse MFA without PCA
t0=cputime;
V_SMFA_PCA=sparse_MFA(x_trn,y_trn,20); 
elapsedtime = cputime - t0;
res(loop,1)= elapsedtime;
%% MFA without PCA
t0=cputime;
[V_MFA,inde] = MFA(x_trn,y_trn,22);
elapsedtime = cputime - t0;
res(loop,2)= elapsedtime;
    
%% Linear discriminant analysis
t0=cputime;
[V_LDA_self,inde] = LDA(x_trn, y_trn,6);
elapsedtime = cputime - t0;
res(loop,3)= elapsedtime;
%% lpp
% t0=cputime;
% [~, V_LPP_ALL,V_LPP,inde] = lpp_self(x_trn);
% elapsedtime = cputime - t0;
% res(loop,4)= elapsedtime;
%% LFDA without PCA  
t0=cputime;
x_trn_col = changeXRow2XCol(x_trn);
[V_LFDA,inde]=LFDA(x_trn_col,y_trn,10); 
elapsedtime = cputime - t0;
res(loop,5)= elapsedtime;       
%% SLFDA
t0=cputime;
x_trn_col = changeXRow2XCol(x_trn);
V_SLFDA=SparseLocal_FDA(x_trn_col,y_trn,24);
elapsedtime = cputime - t0;
res(loop,6)= elapsedtime; 
end
mean(Acc)
% Acc