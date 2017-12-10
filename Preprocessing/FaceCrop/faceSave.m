clear;
stImageFilePath  = 'C:\Users\Andrew\Desktop\SecondPaper-master\';
stImageSavePath  = 'C:\Users\Andrew\Desktop\SecondPaper-master\newPic\';
dirImagePathList = dir(strcat(stImageFilePath,'*.png'));        %��ȡ���ļ���������ͼƬ��·�����ַ�����ʽ��
iImageNum        = length(dirImagePathList);                    %��ȡͼƬ��������
if iImageNum > 0                                                %��������ͼƬ��������ټ�⣬���������
    for i = 1 : iImageNum
        iSaveNum      = int2str(i);
        stImagePath   = dirImagePathList(i).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
        mFaceResult   = face_segment(mImageCurrent);
        imwrite(mFaceResult,strcat(stImageSavePath,iSaveNum,'.bmp')); 
    end 
end