clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ��������ͼ��
x= imread('S010_001_01594215.png');
xx=x;
figure,
imshow(x);


fR=xx(:,:,1);
fG=xx(:,:,2);
fB=xx(:,:,3);
f=1/9*ones(3);%��ͨ�˲������˳���Ƶ����
filtered_fR=imfilter(fR,f);
filtered_fG=imfilter(fG,f);
filtered_fB=imfilter(fB,f);
x_filtered=cat(3,filtered_fR,filtered_fG,filtered_fB);
figure,
imshow(x_filtered);


I=rgb2ycbcr(x);        %��ɫ�ռ�ת�� 
gray=rgb2gray(x);
figure,
imshow(gray);

[a,b,c]=size(I); %�õ�ͼ������ص����
cb=double(I(:,:,2));
cr=double(I(:,:,3));
for i=1:a
    for j=1:b
        w=[cb(i,j),cr(i,j)];
        m=[117.4316 148.5599];
        n=[260.1301 12.1430;12.1430 150.4574];
        p(i,j)=exp((-0.5)*(w-m)*inv(n)*(w-m)');%��ĳ���ص�ĸ���
        if (p(i,j)<0.5) 
            p(i,j)=0;
        else 
            p(i,j)=1;
        end
    end
end 
fenge=p;
figure,
imshow(fenge);

SE = strel('square',3); 
imf=imopen(p,SE);         %�����㣨���ȸ�ʴ�����ͣ���������ɢ�� 
xingtai=imf;
figure,
imshow(xingtai);
%figure,imshow(Ibwopen); 
%Ibwoc=imclose(Ibwopen,SE);      %�����㣬ȥ�����ڿ�������������ȱ�� 
%figure,imshow(Ibwoc); 
%imf=imfill(Ibwoc,'holes');      %���׶� 
%%%%%%%%%%%%%%%%%%%%%%%���������ȥ���ֽš��첲�ȷ���������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L,num]=bwlabel(imf,8);     %��ͨ������ 
B=zeros(size(imf));
for i=1:num
    Area(i)=bwarea(L==i);%����ÿ��Ƥ����������
end
for i=1:num
    [r,c] = find(L==i) ;
    left(i)=min(c);
    right(i)=max(c);
    up(i)=min(r);
    down(i)=max(r);
end
for i=1:num 
    %����������������
    Rect_Area(i)=(down(i)-up(i))*(right(i)-left(i));
end
%���������������
Ratio=Area./Rect_Area;
for i=1:num 
   if Ratio(i)>=0.5%����Ӧ���������ʴ���0.5����������
       [x,y]=find(L==i);%��i�����������ֵ
       B=B+bwselect(imf,y,x,8);%������ʴ���0.5Ƥ�������������   
   end      
end
%%%%%%%%%%%%%%%%%%%%%%%%%%�������������һ����ȥһЩ��С�ķ���������%%%%%%%%%%%%%%%%%%%%%%%%%%
[L1,num1]=bwlabel(B,8);     %��ͨ������
B1=zeros(size(B));
for i=1:num1
    Area(i)=bwarea(L1==i);%����ÿ��Ƥ����������
end
maxarea=max(Area);%ȡ���ֵ
q=Area/maxarea;%ÿ�����������������������ı�ֵ   
for i=1:num1 
   if q(i)>=0.3%����Ӧ����������ֵ����0.3����������
       [x,y]=find(L1==i);%��i�����������ֵ
       B1=B1+bwselect(B,y,x,8);%�������ֵ����0.3Ƥ�������������   
   end      
end
%%%%%%%%%%%%%%%%%%%%%%%%%%���ݷ�ɫ����ĳ��������ȥһЩ����������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L2,num2]=bwlabel(B1,8);     %��ͨ������ 
B2=zeros(size(B1));
for i=1:num2
    [r,c] = find(L2==i);  
    left(i)=min(c);
    right(i)=max(c);
    up(i)=min(r);
    down(i)=max(r);
end
for i=1:num2
    if ((down(i)-up(i))/(right(i)-left(i)))>0.8&((down(i)-up(i))/(right(i)-left(i)))<2
        [x,y]=find(L2==i);
        B2=B2+bwselect(B1,y,x,8);%%%�����㳤�����0.8��2����������
    end
end
figure,
imshow(xx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�����������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L3 num3]=bwlabel(B2,8);     %��ͨ������
for i=1:num3
    [r,c] = find(L3==i);  
    left(i)=min(c);
    right(i)=max(c);
    up(i)=min(r);
    down(i)=max(r);
end

hold on;
for i=1:num3
    if(down(i)>(up(i)+(right(i)-left(i))*1.2))     %�������������
       down(i)=up(i)+(right(i)-left(i))*1.2;
    end
    x=[left(i);left(i);right(i);right(i);left(i)];
    y=[up(i);down(i);down(i);up(i);up(i)];
    plot(x,y);      %����
end
hold off;

a = min(x);b = min(y);
w = max(x) - min(x);
h = max(y) - min(y);
rect  = [a b w h];
%%rect �������ľ����ˡ�
%%Ȼ��ʹ��imcrop�������Խ����������и������
img_cut = imcrop(image,rect);
