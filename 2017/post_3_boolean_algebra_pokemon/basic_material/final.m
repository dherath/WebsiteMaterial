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

figure;
subplot(241);imshow(man);title('charachter (original)');
subplot(242);imshow(man_r);title('charachter (channel R)');
subplot(243);imshow(man_g);title('charachter (channel G)');
subplot(244);imshow(man_b);title('charachter (channel B)');
subplot(245);imshow(back);title('background (original)');
subplot(246);imshow(back_r);title('background (channel R)');
subplot(247);imshow(back_g);title('background (channel G)');
subplot(248);imshow(back_b);title('background (channel B)');

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

% figure;% the images for front matter 
% subplot(131);imshow(back);
% subplot(132);imshow(man);
% subplot(133);imshow(final_image);