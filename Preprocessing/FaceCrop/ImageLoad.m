clear;
stImageFilePath  = 'C:\Users\Andrew\Desktop\SecondPaper-master\Preprocessing\FaceCrop\OriginalDataset\';
stImageSavePath  = 'C:\Users\Andrew\Desktop\SecondPaper-master\Preprocessing\FaceCrop\DividedFace\';
dirImagePathList = dir(strcat(stImageFilePath,'*.png'));        %读取该文件夹下所有图片的路径（字符串格式）
iImageNum        = length(dirImagePathList);                    %获取图片的总数量
if iImageNum > 0                                                %批量读入图片，进行五官检测，再批量检测
    for i = 1 : iImageNum
        iSaveNum      = int2str(i);
        stImagePath   = dirImagePathList(i).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
    end
end