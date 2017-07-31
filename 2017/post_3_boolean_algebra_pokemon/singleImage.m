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

com = and(man_b,not_back);

com1 = not(com);

with_man = com1.*double(man);
with_tree = com.*double(back);

sz= size(com);
for i = 1:sz(1,1)
    for j=1:sz(1,2)
        temp(i,j)=max(with_man(i,j),with_tree(i,j));
    end
end

temp = uint8(temp);
figure;
imshow(temp);