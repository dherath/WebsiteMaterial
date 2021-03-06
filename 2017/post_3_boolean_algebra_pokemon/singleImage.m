clc;
clear;
man_original= imread('silver.png');
back_original = imread('background.png');

man = man_original(:,:,1);%red channel
back=back_original(:,:,1);%red channel

man_b= im2bw(man,0.99);
back_b = im2bw(back);


% figure;
% subplot(141);imshow(man);title('charachter (Red channel)');
% subplot(142);imshow(man_b);title('charachter (Binary)');
% subplot(143);imshow(back);title('background (Red channel)');
% subplot(144);imshow(back_b);title('background (Binary)');


not_back = not(back_b);

% figure;
% subplot(121);imshow(back_b);title('Binary - background');
% subplot(122);imshow(not_back);title('Inverted - background');

com = and(man_b,not_back);
% figure;
% subplot(131);imshow(not_back);title('Inverted-background');
% subplot(132);imshow(man_b);title('Binary-charachter');
% subplot(133);imshow(com);title('Output - Mask 1');


com1 = not(com);
% figure;
% subplot(121);imshow(com);title('Input - Mask 1');
% subplot(122);imshow(com1);title('Output - Mask 2');


with_man = com1.*double(man);
with_tree = com.*double(back);

% figure;
% subplot(231);imagesc(com1);colormap(gray);axis off;title('Mask 2');
% subplot(232);imagesc(man);axis off;title('Charachter (Red channel)');
% subplot(233);imagesc(with_man);axis off;title('Charachter output');
% subplot(234);imagesc(com);axis off;title('Mask 1');
% subplot(235);imagesc(back);axis off;title('Background (Red channel)');
% subplot(236);imagesc(with_tree);axis off;title('Background output');



sz= size(com);
for i = 1:sz(1,1)
    for j=1:sz(1,2)
        temp(i,j)=max(with_man(i,j),with_tree(i,j));
    end
end
% figure;
% subplot(131);imagesc(with_tree);colormap(gray);axis off;title('Background output');
% subplot(132);imagesc(with_man);colormap(gray);axis off; title('Charachter ouput');
% subplot(133);imagesc(temp);colormap(gray);axis off;title('Combined Image (R)');

temp = uint8(temp);
% figure;
% imshow(temp);