clc;
clear;
man = imread('silver.png');
back = imread('background.png');

man_r=man(:,:,1);
man_g=man(:,:,2);
man_b=man(:,:,3);

back_r = back(:,:,1);
back_g = back(:,:,2);
back_b = back(:,:,3);

temp_r = combine_image(man_r,back_r);
temp_g = combine_image(man_g,back_g);
temp_b = combine_image(man_b,back_b);

figure;
subplot(131);imshow(temp_r);axis image;
subplot(132);imshow(temp_g);axis image;
subplot(133);imshow(temp_b);axis image;

final_image = uint8(zeros(size(man)));
final_image(:,:,1)= temp_r;
final_image(:,:,2)= temp_g;
final_image(:,:,3)= temp_b;

figure;
imshow(final_image);