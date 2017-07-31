clc;
clear;
man = imread('silver.png');
back = imread('background.png');

man_r=man(:,:,1);
man_g=man(:,:,2);
man_b=man(:,:,3);

back_r = back(:,:,1);

man_level = graythresh(man_r);
man_rb= im2bw(man_r,0.999);

back_rb = im2bw(back_r);

% com = not(and(man_rb,back_rb));
% 
% with_man = com.*double(man_r);
% with_trees = com.*double(back_r);
% imagesc(with_trees);axis image;
not_back = not(back_rb);
com = and(man_rb,not_back);
com1 = not(com);
% imtool(com);
% imtool(com1);

with_man = com1.*double(man_r);
% imtool(with_man);
with_tree = com.*double(back_r);
% imtool(with_tree);
% subplot(121);imagesc(with_man);axis image;
% subplot(122);imagesc(with_tree);axis image;
sz= size(com);
for i = 1:sz(1,1)
    for j=1:sz(1,2)
        temp(i,j)= max(with_man(i,j),with_tree(i,j));
    end
end

figure;
subplot(131);imagesc(man_r);axis image;
subplot(132);imagesc(back_r);axis image;
subplot(133);imagesc(temp);axis image;