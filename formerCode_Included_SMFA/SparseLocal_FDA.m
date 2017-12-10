function V=SparseLocal_FDA(XXX,Y,r,kNN,epsilon)
%
% ϡ��ֲ�Fisher�б���������мල��ά��
%       
% ����:
%    X:      d �� n ԭʼ��������     
%            d --- ������ά�ȣ�jaffe����Ϊ64*64��
%            n --- �����ĸ�����jaffeΪ213��
%    Y:      n ά������ǩ��  
%    r:      ��ά�ռ��ά�ȣ�Ĭ��ֵd��
%    kNN:    ����ֲ������ڽ�Ȩֵ����ʱҪ�õ�������ڲ���(Ĭ��ֵ 7)
%    epsilon:torlence  
% ���:
%    V: d �� r ͶӰ���� (Z=T'*X)
%    Z: r �� n ��ά����������� 

XXX = double(XXX);

[d,n]=size(XXX);

if nargin<4
%    kNN=7;
  kNN=1;
end

if nargin<3
  r=d;
end

tSb=zeros(d,d);%�����ɢ�Ⱦ���
tSw=zeros(d,d);%������ɢ�Ⱦ���
W = ones(n)./n;
for c=1:max(Y)
  Xc=XXX(:,Y==c);
  Temp = find(Y==c);              
  nc=size(Xc,2); % ����Xc����������nc

  % ����Ȩֵ����A  LFDA���þֲ�����Ȩֵ����
  Xc2=sum(Xc.^2,1);
  distance2=repmat(Xc2,nc,1)+repmat(Xc2',1,nc)-2*Xc'*Xc;
  [sorted,index]=sort(distance2);
  kNNdist2=sorted(kNN+1,:);
  sigma=sqrt(kNNdist2);
  localscale=sigma'*sigma;
  flag=(localscale~=0);
  A=zeros(nc,nc);
  A(flag)=exp(-distance2(flag)./localscale(flag));
   
  W(Temp,Temp)=A./n;
  
  Xc1=sum(Xc,2);
  G=Xc*(repmat(sum(A,2),[1 d]).*Xc')-Xc*A*Xc';
  tSb=tSb+G/n+Xc*Xc'*(1-nc/n)+Xc1*Xc1'/n;
  tSw=tSw+G/nc;
end

%��������������ɢ�Ⱦ���
X1=sum(XXX,2);
tSb=tSb-X1*X1'/n-tSw;

tSb=(tSb+tSb')/2;
tSw=(tSw+tSw')/2;

Dlt = diag(sum(W));
Llt = Dlt - W;
[Qlt,Alpha]=eig(Llt);
Hlt = XXX*Qlt*abs(Alpha^(1/2));
[P,aaa,~]=svd(Hlt);  
index=find(diag(aaa)>1.e-5);
P=P(:,index);
                                                                                    
Sb = P'*tSb*P;
Sw = P'*tSw*P;
[U,D]=eig(Sb,Sw);
[D,index]=sort(diag(D));%����LΪ�ԽǵĶԽǾ���
U=U(:,index);
index=find(D>1.e-5);%�ҵ�����0�ĸþ�����±�

D=D(index(end-r+1:end));
U=U(:,index(end-r+1:end));%�����������ֵ����Ӧ����������

%����bregman��������ϡ�账��
% V = P*U;
tq = size(U,2);
%��ʼ����ر���
% [n,d]=size(XXX);
V = zeros(d,tq); B = zeros(d,tq);                                                                                         
delta=.01; mu=.5;    % delta mu ��Ϊ���� �Աȵ����õ���V��PU��˵�V
if nargin<6
    %epsilon=10^(-5);
    epsilon=1;
end
%���Ե����㷨
k=1;
ttemp(1) = norm(P'*V-U);
while ttemp(k)>=epsilon && k<=10000 %ѭ���ж�
    k=k+1;
    B = B - P*(P'*V-U);
    V = delta*shrink(B,mu);
    ttemp(k) = norm(P'*V-U);
end

% Z=V'*XXX;X_tst = V'*x_tst;
