%===============================================================================
%�������ƣ�face_segment
%���������mImageSrc�����ָ������ͼ�񣬿����ǻҶ�ͼ��Ҳ�����ǲ�ɫͼ��
%���������mFaceResult���ָ������������ӦΪ�Ҷ�ͼ��
%��Ҫ���裺1������������⣬�õ���������Ŀ��
%         2���õ�����ͼ�������ĵ�
%         3���������ĵ㣬��ͼ����еȱ����������õ����ʴ�С������ͼ��
%ע�����1)������Ҫ�жϸ�ͼ���Ƿ�Ϊ�Ҷ�ͼ����Ϊ�Ҷ�ͼ����Ҫ�Ƚ���ת��Ϊ��ͨ����ɫͼ
%===============================================================================
function mFaceResult = face_segment(mImageSrc)
%%%%%%%%%%%%%%%%%%%%���Ҷ�ͼ��Ϊ��ͨ��ͼ%%%%%%%%%%%%%%%%%%%%
if(size(mImageSrc,3) == 1)
    mImage2detect(:,:,1) = mImageSrc;
    mImage2detect(:,:,2) = mImageSrc;
    mImage2detect(:,:,3) = mImageSrc;
else
    mImage2detect = mImageSrc;
end

%%%%%%%%%%%%%%%%%%%%��ͼ������������%%%%%%%%%%%%%%%%%%%%
FaceDetector               = buildDetector();
[bbox,bbimg,faces,bbfaces] = detectFaceParts(FaceDetector,mImage2detect,2);

%%%%%%%%%%%%%%%%%%%%����ͼ��ҶȻ�%%%%%%%%%%%%%%%%%%%%
if 1 ~= size(mImageSrc,3)
    mImageSrc = rgb2gray(mImageSrc);
    mImageSrc = double(mImageSrc);
elseif 1     == size(mImageSrc,3)
    mImageSrc = double(mImageSrc);
end

%%%%%%%%%%%%%%%%%%%%�õ��������������ĵ�%%%%%%%%%%%%%%%%%%%%
recFace.x          = bbox(1,1);
recFace.y          = bbox(1,2);
recFace.width      = bbox(1,3);
recFace.height     = bbox(1,4);

ptFaceCenter.x     = recFace.x + recFace.width / 2;
ptFaceCenter.y     = recFace.y + recFace.height / 2;

%%%%%%%%%%%%%%%%%%%%�����ĵ�Ϊ��׼������������������ѡ����е�����%%%%%%%%%%%%%%%%%%%%
recFace.x         = ptFaceCenter.x - recFace.width * 0.4;
recFace.y         = ptFaceCenter.y - recFace.height * 0.35;
recFace.width     = recFace.width * 0.8 ;
recFace.height    = recFace.height * 0.8 ;

mFaceResult       = uint8(imcrop(mImageSrc,[recFace.x,recFace.y,recFace.width,recFace.height]));
end