clc;
clear;

% man = imread('manf.png');
% %imtool(man);
% back = imread('background.png');
% sz= size(back);
% man1 = imresize(man,[sz(1,1) sz(1,2)]);
% 
% imwrite(man1,'silver.png');

man = imread('silver.png');
back = imread('background.png');

man_blkwht = im2bw(man);
back_blkwht = im2bw(back);
combined1 = and(man_blkwht,back_blkwht);

com1 = im2double(combined1);

man_r = im2double(man(:,:,1));
man_g= im2double(man(:,:,2));
man_b= im2double(man(:,:,3));
% figure;
% subplot(131);imshow(man_r);
% subplot(132);imshow(man_b);
% subplot(133);imshow(man_g);
combined = im2double(not(combined1));

img = uint8(zeros(size(man)));
img(:,:,1)= or(man_r,combined);
img(:,:,2)= or(man_g,combined);
img(:,:,3)= or(man_b,combined);
imagesc(img);
% 
% figure;
% subplot(131);imshow(man_blkwht);
% subplot(132);imshow(back_blkwht);
% subplot(133);imshow(combined);




