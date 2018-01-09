function [mapping,index] = MFA(X_trn,Y_trn,dims)
% MPA Perform Marginal Fisher Analysis
% Perform the Marginal Fisher Analysis on dataset X_trn to reduce it 
% dimensionality no_dims,The number fo neighbors that si used by MFA is
% specified by k=1;
% original author @Tomek 
% if ~exist('no_dims','var')
%     no_dims=2;
% end
X_trn = double(X_trn);
F1=zeros(size(X_trn,1),size(X_trn,1));
for i=1:size(X_trn,1)
      for j=1:size(X_trn,1)
           dist(i,j)=norm(X_trn(i,:)-X_trn(j,:),2);
      end
end
[tmp,ind]=sort(dist,2);
% 构建邻接图F1
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
% 构建邻接图F2
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

Sw=X_trn'*(S1-F1)*X_trn;
Sb=X_trn'*(S2-F2)*X_trn;
% Sw=Sw+eye(size(Sw,1))*1.e-8;
% Sb=Sb+eye(size(Sb,1))*1.e-8;

Sb=(Sb+Sb')/2;
Sw=(Sw+Sw')/2;

Sw=Sw+eye(size(Sw,1))*1.e-8;

S=inv(Sw)*Sb;
S=(S+S')/2;
[V, L] = eig(S);

% [V,L]=eig(Sb,Sw);

[L,index]=sort(diag(L));
mapping=V(:,index);
% index=find(L>1.e-5);
% V=V(:,index);
index=find(L>1.e-5);

L=L(index(end-dims+1:end));
mapping=V(:,index(end-dims+1:end));

% [eigvectors,eigvalues]=eig(Sw,Sb);
% index=find(diag(eigvalues)>1.e-5);     %加上之后准确度下降约10%
% eigvectors=eigvectors(:,index);

% [eigvalues,index]=sort(diag(eigvalues));
% eigvectors=eigvectors(:,index);
% index=find(eigvalues>1.e-5);
% 
% eigvalues=eigvalues(index(end-dims+1:end));
% mapping=eigvectors(:,index(end-dims+1:end));
%mapping=uint8(mapping);
% XX_trn=X_trn*mapping;
%  XX_tst=X_tst*mapping;


