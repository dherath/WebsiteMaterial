clc;
clear;

im1 = imread('circles.png');
im2 = not(im1);

figure;
subplot(121);imshow(im1);title('original image(Im1)');
subplot(122);imshow(im2);title('NOT(Im1)');