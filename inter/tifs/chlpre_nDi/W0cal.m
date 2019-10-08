function [ W0 ] = W0cal( V, Wi, Y, param )

    parta1 = zeros(size(V{1},1),size(V{1},1));
    partb = zeros(size(V{1},1),1);
    partb1 = zeros(size(V{1},1),1);
    partb2 = zeros(size(V{1},1),1);
    

    for i = 1 : length(V)

        Vi = V{i};
        Wione = Wi(:,i);
        ViViT = Vi*Vi';
        
        parta1 = parta1 + ViViT;
        
        partb1 = Vi*Y';
        partb2 = ViViT*Wione;
        
        partb = partb + (partb1-partb2);
    end

    parta = parta1 + param.nata*eye(size(parta1,1));
    W0 = parta\partb;

end

