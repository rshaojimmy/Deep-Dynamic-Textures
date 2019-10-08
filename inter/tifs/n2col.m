function [ featnor ] = n2col( feat )

    for i = 1 : size(feat,2)
        featnor(:,i) = feat(:,i) /norm(feat(:,i));
    end
    
end

