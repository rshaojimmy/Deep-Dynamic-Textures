function [ W0 ] = W0calnew( V, Wi, D, Y, param )

    parta1 = zeros(size(V{1},1),size(V{1},1));
    partb = zeros(size(V{1},1),1);
    partb1 = zeros(size(V{1},1),1);
    partb2 = zeros(size(V{1},1),1);
    
    N = length(Y);
    
    for i = 1 : length(V)
        Di = D(i);
        Vi = V{i};
        Wione = Wi(:,i);
        ViViT = Vi*Vi';
        
        parta1 = parta1 + Di*ViViT;
        
        partb1 = Di*Vi*Y';
        partb2 = ViViT*Wione;
        
        partb = partb + (partb1-partb2);
    end

    parta = N.^(-1)* parta1 + param.nata*eye(size(parta1,1));
    W0 = parta\(N.^(-1)*partb);

end

