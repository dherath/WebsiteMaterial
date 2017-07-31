clc;
clear;

im1 = imread('circles.png');
im2 = not(im1);

figure;
subplot(121);imshow(im1);title('input');
subplot(122);imshow(im2);title('output');