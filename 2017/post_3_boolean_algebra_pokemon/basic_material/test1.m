clc;
clear
man = imread('silver.png');
back = imread('background.png');

man_bw =im2bw(man);
back_bw = im2bw(back);
not_man = not(man_bw);
% figure;
% subplot(121);imshow(man_bw);
% subplot(122);imshow(not_man);

% image = and(not_man,back_bw);
% figure; imshow(image);

red_channel = and(man_bw,back(:,:,1));
blue_channel = and(man_bw,back(:,:,2));
grn_channel = and(man_bw,back(:,:,3));

rd = man_bw.*back(:,:,1);
b=man_bw.*back(:,:,2);
grn= man_bw.*back(:,:3);


comp = zeros(size(man));
c1 = comp;
comp(:,:,1)=red_channel;
comp(:,:,2)=blue_channel;
comp(:,:,3)=grn_channel;
figure;imshow(comp);

c1(:,:,1)=rd;
c1(:,:,2)=b;
c1(:,:,3)=grn;
figure;imshow(c1);


