function temp = combine_image(man,back)
    man_b= im2bw(man,0.999);

    back_b = im2bw(back);

    not_back = not(back_b);

    com = and(man_b,not_back);

    com1 = not(com);

    with_man = com1.*double(man);
    with_tree = com.*double(back);

    sz= size(com);
    for i = 1:sz(1,1)
        for j=1:sz(1,2)
            temp(i,j)= max(with_man(i,j),with_tree(i,j));
        end
    end
    
end

