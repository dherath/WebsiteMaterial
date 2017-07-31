clc;
clear;

img1 = imread('circles.png');
img2 = imread('circles2.png');

img1= im2bw(img1);
img2=im2bw(img2);

img_and = and(img1,img2);
figure;
subplot(131);imshow(img1);title('input 1');
subplot(132);imshow(img2);title('input 2');
subplot(133);imshow(img_and);title('output');

img_or = or(img1,img2);
figure;
subplot(131);imshow(img1);title('input 1');
subplot(132);imshow(img2);title('input 2');
subplot(133);imshow(img_or);title('output');