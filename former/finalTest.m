clear;
load Jaffe;
load Jaffe32_row;
% load CK32RowFinal.mat;
% load CKROW_Y.mat;
global p1 knn
knn=1;
p1=2^-8;
X = Jaffe32_row;
Y=Y;
for loop=1:20
   index1=find(Y==1);
   index2=find(Y==2);
   index3=find(Y==3);
   index4=find(Y==4);
   index5=find(Y==5);
   index6=find(Y==6);
   index7=find(Y==7); 
     
   index1 = index1(randperm(length(index1)));
   index2 = index2(randperm(length(index2)));
   index3 = index3(randperm(length(index3)));
   index4 = index4(randperm(length(index4)));
   index5 = index5(randperm(length(index5)));
   index6 = index6(randperm(length(index6)));
   index7 = index7(randperm(length(index7)));
   
   sampN= 12;
   in1 = index1(1:sampN,1);
   in2 = index2(1:sampN,1);
   in3 = index3(1:sampN,1);
   in4 = index4(1:sampN,1);
   in5 = index5(1:sampN,1);
   in6 = index6(1:sampN,1);
   in7 = index7(1:sampN,1);
   
   trn = zeros(213,1);
   trn(in1(:,1)) = 1;
   trn(in2(:,1)) = 1;
   trn(in3(:,1)) = 1;
   trn(in4(:,1)) = 1;
   trn(in5(:,1)) = 1;
   trn(in6(:,1)) = 1;
   trn(in7(:,1)) = 1;
   
   ttrn = logical(trn);
   ttst = ~ttrn;
   x_trn=X(ttrn,:);
   y_trn=Y(ttrn);
   x_tst=X(ttst,:);
   y_tst=Y(ttst);
   
    [eigvectorPCA,eigvaluePCA] = PCA2(double(x_trn),1003);
    x_trn_PCA = double(x_trn)*eigvectorPCA;
    x_tst_PCA = double(x_tst)*eigvectorPCA; 

    %% sparse MFA with PCA
%     for i=1:20
%         V_SMFA_PCA=sparse_MFA(x_trn_PCA,y_trn,i); 
%         X_trn=x_trn_PCA*V_SMFA_PCA;
%         X_tst=x_tst_PCA*V_SMFA_PCA;
%         [out1]=cknear(knn,X_trn,y_trn,X_tst); 
%         res1(loop,i)=mean(out1==y_tst);
%     end

    %% sparse MFA without PCA
%      for i=1:20
%         V_SMFA_PCA=sparse_MFA(x_trn,y_trn,i); 
%         x_trn = double(x_trn);
%         x_tst= double(x_tst);
%         X_trn=x_trn*V_SMFA_PCA;
%         X_tst=x_tst*V_SMFA_PCA;
%         [out2]=cknear(knn,X_trn,y_trn,X_tst); 
%         res2(loop,i)=mean(out2==y_tst);
%      end

    %% MFA with PCA
%     [V_MFA_PCA,inde] = MFA(x_trn_PCA,y_trn);
%     for i=1:20
%         x_trn = double(x_trn);
%         x_tst= double(x_tst);
%         mapping=V_MFA_PCA(:,inde(end-i+1:end));
%         X_trn=x_trn_PCA*mapping;
%         X_tst=x_tst_PCA*mapping;
%         [out3]=cknear(knn,X_trn,y_trn,X_tst); 
%         res3(loop,i) = mean(out3==y_tst);
%     end

    %% MFA without PCA
%     [V_MFA,inde] = MFA(x_trn,y_trn);
%     for i=1:20
%         x_trn = double(x_trn);
%         x_tst= double(x_tst);
%         mapping=V_MFA(:,inde(end-i+1:end));
%         X_trn=x_trn*mapping;
%         X_tst=x_tst*mapping;
%         [out4]=cknear(knn,X_trn,y_trn,X_tst); 
%         res4(loop,i) = mean(out4==y_tst);
%     end
    
%% Linear discriminant analysis
x_trn = double(x_trn);
[V_LDA_self,inde] = LDA(x_trn, y_trn,3);
    for i=1:3
        x_trn = double(x_trn);
        x_tst= double(x_tst);
        mapping = V_LDA_self(:,inde(end-i+1:end));
        X_trn=x_trn*mapping;
        X_tst=x_tst*mapping;
        [out5]=cknear(knn,X_trn,y_trn,X_tst); 
        res5_CKonly6_12(loop,i) = mean(out5==y_tst);
    end

     %% lpp
%     x_trn=double(x_trn);
%     x_tst=double(x_tst);
%     [~, V_LPP_ALL,V_LPP,inde] = lpp_self(x_trn_PCA);
%     for i=1:20
%         mapping = V_LPP(:,inde(1:i));
%         X_trn=x_trn_PCA*mapping;
%         X_tst=x_tst_PCA*mapping;
%         [out6]=cknear(knn,X_trn,y_trn,X_tst); 
%         res6(loop,i)=mean(out6==y_tst);
%     end
    %% LFDA without PCA  
    x_trn_col = changeXRow2XCol(x_trn);
    x_tst_col = changeXRow2XCol(x_tst);
    [V_LFDA,inde]=LFDA(x_trn_col,y_trn);
    for i=1:20
        mapping = V_LFDA(:,inde(end-i+1:end));
        x_newtst=mapping'*x_tst_col;
        x_newtrn=mapping'*x_trn_col;
        X_trn = changeXCol2XRow(x_newtrn);
        X_tst = changeXCol2XRow(x_newtst);
        [out7]=cknear(knn,X_trn,y_trn,X_tst);
        res7(loop,i)=mean(out7==y_tst);
    end  
        
    %% SLFDA
    x_trn_col = changeXRow2XCol(x_trn);
    x_tst_col = changeXRow2XCol(x_tst);
    for i=1:20
        V_SLFDA=SparseLocal_FDA(x_trn_col,y_trn,i);
        X_trn=V_SLFDA(:,1:i)'*x_trn_col;
        X_tst = V_SLFDA(:,1:i)'*x_tst_col;
        X_trn = changeXCol2XRow(X_trn);
        X_tst = changeXCol2XRow(X_tst);
        [out8]=cknear(knn,X_trn,y_trn,X_tst); 
        res8(loop,i)=mean(out8==y_tst);
    end
    
end


% % % plot(1:20,mean(res3),'green');
% % % hold on;
% % plot(1:20,mean(res4),'yellow');
% % hold on;
% % plot(1:20,mean(res5_CKonly6_12),'red');
% % hold on;
% % plot(1:20,mean(res6),'cyan');
% % hold on;
% % plot(1:20,mean(res7),'magenta');
% % hold on;
% % plot(1:20,mean(res8));
% % hold on;
% % % plot(1:20,mean(res1),'black');
% % % hold on;
% % plot(1:20,mean(res2),'blue');
% % hold on;
% % legend('MFA','LDA','LPP','LFDA','SLFDA','SMFA');
% % xlabel('Dimension');
% % ylabel('Recognition rate');
















% A = mean(Acc);
% A = squeeze(A);
% plot(A(:,1),'black')
% hold on;
%  axis([2 40 0 0.9])
% plot(A(:,2),'blue')
% hold on;
% x = find(A(:,3)>0);
% [d1,d2] = size(x);
% plot(A(1:d1-1,3))
% hold on;
% x = find(A(:,4)>0);
% [d1,d2] = size(x);
% plot(A(1:d1,4))
% hold on;
% x = find(A(:,5)>0);
% [d1,d2] = size(x);
% plot(A(1:d1-1,5))
% hold on;
% plot(A(:,6))
% hold on;
% x = find(A(:,7)>0);
% [d1,d2] = size(x);
% plot(A(1:d1-1,7))
% hold on;
% x = find(A(:,8)>0);
% [d1,d2] = size(x);
% plot(A(1:d1,8))
% hold on;
% % legend('SMFA+PCA','SMFA','MFA+PCA','MFA','LDA','LPP');
% legend('SMFA+PCA','SMFA','MFA+PCA','MFA','LDA','LPP','LFDA','SLFDA');
% xlabel('Dimension');
% ylabel('Recognition rate');






% 
%    x=2:2:size(V_SMFA_PCA,2);
%     plot(x,mean(Acc(loop,1:size(x,2),1)),'b')
%     hold on;
%     axis([2 40 0 0.9])
%     x=2:2:size(V_SMFA,2);
%     plot(x,Acc(loop,1:size(x,2),2),'c')
%     hold on;
%     x=2:2:size(V_MFA_PCA,2);
%     plot(x,Acc(loop,1:size(x,2),3),'red')
%     hold on;
%     x=2:2:size(V_MFA,2);
%     plot(x,Acc(loop,1:size(x,2),4),'yellow')
%     hold on;
%     x=2:2:size(V_LDA_self,2);
%     plot(x,Acc(loop,1:size(x,2),5),'black')
%     hold on;
%     x=2:2:size(V_LPP_SELF,2);
%     plot(x,Acc(loop,1:size(x,2),6),'green')
%     hold on;
%     x=2:2:size(V_LFDA,2);
%     plot(x,Acc(loop,1:size(x,2),7),'cyan')
%     hold on;   
%     x=2:2:size(V_SLFDA,2);
%     plot(x,Acc(loop,1:size(x,2),8),'magenta')
%     hold off;       


% mean(Acc)
% std(Acc)