function [mapping,PP] = sparse_MFA(X_trn,Y_trn,dims)
% MPA Perform Marginal Fisher Analysis
% Perform the Marginal Fisher Analysis on dataset X_trn to reduce it 
% dimensionality no_dims,The number fo neighbors that si used by MFA is
% specified by k=1;
% original author @Tomek 
% if ~exist('no_dims','var')
%     no_dims=2;
% end
%X:һ��һ������

X_trn = double(X_trn);

F1=zeros(size(X_trn,1),size(X_trn,1));
for i=1:size(X_trn,1)
      for j=1:size(X_trn,1)
           dist(i,j)=norm(X_trn(i,:)-X_trn(j,:),2);
      end
end
[tmp,ind]=sort(dist,2);
% �����ڽ�ͼF1
for i=1:size(X_trn,1)
    for j=(i+1):size(X_trn,1)
        if Y_trn(i)==Y_trn(ind(i,j));
            F1(i,ind(i,j))=1;
            F1(ind(i,j),i)=1;
            break;
        else
            continue;
        end
    end
end
% �����ڽ�ͼF2
F2=zeros(size(X_trn,1),size(X_trn,1));
for i=1:size(X_trn,1)
    for j=(i+1):size(X_trn,1)
        if Y_trn(i)~=Y_trn(ind(i,j));
            F2(i,ind(i,j))=1;
            F2(ind(i,j),i)=1;
            break;
        else
            continue;
        end
    end
end

S1=diag(sum(F1));
S2=diag(sum(F2));

F=F1+F2;
Dlt=diag(sum(F));
Llt=Dlt-F;
[Qlt,Alpha]=eig(Llt);
Hlt=X_trn'*Qlt*abs(Alpha^(1/2));
[P,aaa,~]=svd(Hlt); 
index=find(diag(aaa)>1.e-5);
P=P(:,index);

Sw=(X_trn*P)'*(S1-F1)*(X_trn*P);  %(x_trn*PP)'*(S-F)*(x_trn*PP);
Sb=(X_trn*P)'*(S2-F2)*(X_trn*P);
Sw=Sw+eye(size(Sw,1))*1.e-8;
Sb=Sb+eye(size(Sw,1))*1.e-8;
[U,D]=eig(Sb,Sw);
[D,index]=sort(diag(D));%����LΪ�ԽǵĶԽǾ���
U=U(:,index);
index=find(D>1.e-5);%�ҵ�����0�ĸþ�����±�

D=D(index(end-dims+1:end));
U=U(:,index(end-dims+1:end));%�����������ֵ����Ӧ����������

%����bregman��������ϡ�账��
tq = size(U,2);
%��ʼ����ر���
[n,d]=size(X_trn);
V = zeros(d,tq); B = zeros(d,tq);    
delta=.01; mu=0.5;    % delta mu ��Ϊ���� �Աȵ����õ���V��PU��˵�V
% delta=.01; mu=1; i   % delta mu ��Ϊ���� �Աȵ����õ���V��PU��˵�V
epsilon=10^(-10);
%���Ե����㷨
k=1;
ttemp(1) = norm(P'*V-U);
% ttemp(1) = norm(PP'*V-U)/norm(U,2);
while ttemp(k)>=epsilon && k<10000 %ѭ���ж� ���ϵ�������
    k=k+1;
    B = B - P*(P'*V-U);
    V = delta*shrink(B,mu);
    ttemp(k) = norm(P'*V-U);   
end

PP=U;
mapping=V;


