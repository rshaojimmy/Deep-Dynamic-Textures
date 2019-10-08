function [ featnor ] = n2row( feat )

    for i = 1 : size(feat,1)
        featnor(i,:) = feat(i,:)./norm(feat(i,:));
    end
    
end

