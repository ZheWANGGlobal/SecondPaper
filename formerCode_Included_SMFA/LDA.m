function [W,index]=LDA(Input,Target,dims)
% Input:    n*d matrix,each row is a sample;
% Target:   n*1 matrix,each is the class label 
% W:        d*(k-1) matrix,to project samples to (k-1) dimention
% cneters:  k*(k-1) matrix,the means of each after projection 


% 初始化
[n dim]=size(Input);
ClassLabel=unique(Target);
k=length(ClassLabel);

nGroup=NaN(k,1);            % group count
GroupMean=NaN(k,dim);       % the mean of each value
W=NaN(k-1,dim);             % the final transfer matrix
SB=zeros(dim,dim);          % 类间离散度矩阵
SW=zeros(dim,dim);          % 类内离散度矩阵

% 计算类内离散度矩阵和类间离散度矩阵
for i=1:k    
    group=(Target==ClassLabel(i));
    nGroup(i)=sum(double(group));
    GroupMean(i,:)=mean(Input(group,:));
    tmp=zeros(dim,dim);
    for j=1:n
        if group(j)==1
            t=Input(j,:)-GroupMean(i,:);
            tmp=tmp+t'*t;
        end
    end
    SW=SW+tmp;
end
m=mean(GroupMean);    
for i=1:k
    tmp=GroupMean(i,:)-m;
    SB=SB+nGroup(i)*tmp'*tmp;
end

% [V, L] = eig(SB,SW);
% [L,index]=sort(diag(L));
% index=find(L>1.e-5);
% W=V(:,index);

SB=(SB+SB')/2;
SW=(SW+SW')/2;

SW=SW+eye(size(SW,1))*1.e-8;
S=inv(SW)*SB;
S=(S+S')/2;
[V, L] = eig(S);
[L,index]=sort(diag(L));
W=V(:,index);
index=find(L>1.e-5);


% L=L(index(end-6+1:end));
% W=V(:,index(end-6+1:end));

L=L(index(end-dims+1:end));
W=V(:,index(end-dims+1:end));


    
    