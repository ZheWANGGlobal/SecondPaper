function CK32RowFinal=changeCK64Row2CK32Row()
load CK64Row;
CK32RowFinal = zeros(314,32*32);
for i=1:314   % ∂¡»°X_trn  
    amat  = CK64Row(i,:);
     pic = reshape(amat,64,64);
%     imshow(pic);
    newPic = imresize(pic,[32,32]);
%     imshow(newPic);
    aline = reshape(newPic,1,32*32);
    pic = reshape(aline,32,32);
    imshow(pic);
    CK32RowFinal(i,:) = aline;
    aline = CK32RowFinal(i,:);
    pic = reshape(aline,32,32);
    
    imshow(pic);
end
% save CK32RowFinal.mat CK32RowFinal;
end