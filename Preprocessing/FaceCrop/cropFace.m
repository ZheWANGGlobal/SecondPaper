clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 读入待检测图像
x= imread('S010_001_01594215.png');
xx=x;
figure,
imshow(x);


fR=xx(:,:,1);
fG=xx(:,:,2);
fB=xx(:,:,3);
f=1/9*ones(3);%低通滤波器，滤除高频噪声
filtered_fR=imfilter(fR,f);
filtered_fG=imfilter(fG,f);
filtered_fB=imfilter(fB,f);
x_filtered=cat(3,filtered_fR,filtered_fG,filtered_fB);
figure,
imshow(x_filtered);


I=rgb2ycbcr(x);        %颜色空间转换 
gray=rgb2gray(x);
figure,
imshow(gray);

[a,b,c]=size(I); %得到图像的像素点个数
cb=double(I(:,:,2));
cr=double(I(:,:,3));
for i=1:a
    for j=1:b
        w=[cb(i,j),cr(i,j)];
        m=[117.4316 148.5599];
        n=[260.1301 12.1430;12.1430 150.4574];
        p(i,j)=exp((-0.5)*(w-m)*inv(n)*(w-m)');%算某象素点的概率
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
imf=imopen(p,SE);         %开运算（即先腐蚀再膨胀），消除杂散点 
xingtai=imf;
figure,
imshow(xingtai);
%figure,imshow(Ibwopen); 
%Ibwoc=imclose(Ibwopen,SE);      %闭运算，去掉由于开运算引入的许多缺口 
%figure,imshow(Ibwoc); 
%imf=imfill(Ibwoc,'holes');      %填充孔洞 
%%%%%%%%%%%%%%%%%%%%%%%根据填充率去除手脚、胳膊等非人脸区域%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L,num]=bwlabel(imf,8);     %连通区域标记 
B=zeros(size(imf));
for i=1:num
    Area(i)=bwarea(L==i);%计算每个皮肤区域的面积
end
for i=1:num
    [r,c] = find(L==i) ;
    left(i)=min(c);
    right(i)=max(c);
    up(i)=min(r);
    down(i)=max(r);
end
for i=1:num 
    %计算各矩形区域面积
    Rect_Area(i)=(down(i)-up(i))*(right(i)-left(i));
end
%计算各区域的填充率
Ratio=Area./Rect_Area;
for i=1:num 
   if Ratio(i)>=0.5%若相应区域的填充率大于0.5则保留该区域
       [x,y]=find(L==i);%第i块区域的坐标值
       B=B+bwselect(imf,y,x,8);%把填充率大于0.5皮肤区域叠加起来   
   end      
end
%%%%%%%%%%%%%%%%%%%%%%%%%%根据面积比来进一步除去一些较小的非人脸区域%%%%%%%%%%%%%%%%%%%%%%%%%%
[L1,num1]=bwlabel(B,8);     %连通区域标记
B1=zeros(size(B));
for i=1:num1
    Area(i)=bwarea(L1==i);%计算每个皮肤区域的面积
end
maxarea=max(Area);%取最大值
q=Area/maxarea;%每块区域的面积与最大区域面积的比值   
for i=1:num1 
   if q(i)>=0.3%若相应区域的面积比值大于0.3则保留该区域
       [x,y]=find(L1==i);%第i块区域的坐标值
       B1=B1+bwselect(B,y,x,8);%把面积比值大于0.3皮肤区域叠加起来   
   end      
end
%%%%%%%%%%%%%%%%%%%%%%%%%%根据肤色区域的长宽比来除去一些非人脸区域%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L2,num2]=bwlabel(B1,8);     %连通区域标记 
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
        B2=B2+bwselect(B1,y,x,8);%%%把满足长宽比在0.8到2的区域留下
    end
end
figure,
imshow(xx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%把人脸框出来%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[L3 num3]=bwlabel(B2,8);     %连通区域标记
for i=1:num3
    [r,c] = find(L3==i);  
    left(i)=min(c);
    right(i)=max(c);
    up(i)=min(r);
    down(i)=max(r);
end

hold on;
for i=1:num3
    if(down(i)>(up(i)+(right(i)-left(i))*1.2))     %人脸长宽比限制
       down(i)=up(i)+(right(i)-left(i))*1.2;
    end
    x=[left(i);left(i);right(i);right(i);left(i)];
    y=[up(i);down(i);down(i);up(i);up(i)];
    plot(x,y);      %画框
end
hold off;

a = min(x);b = min(y);
w = max(x) - min(x);
h = max(y) - min(y);
rect  = [a b w h];
%%rect 就是最后的矩形了。
%%然后使用imcrop函数可以将矩形区域切割出来。
img_cut = imcrop(image,rect);
