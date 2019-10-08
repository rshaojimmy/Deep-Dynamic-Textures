function [ Wi ] = Wicalnew( V, W0, D, Y, param )

    Wi = zeros(size(V{1},1), length(V));
    parta = zeros(size(V{1},1),size(V{1},1));
    partb = zeros(size(V{1},1),1); 

    N = length(Y);
    for i = 1 : length(V)
        Di = D(i);
        Vi = V{i};
        ViViT = Vi*Vi';
               
        parta = (N*Di).^(-1)*ViViT + (param.beta * Di.^(-2)+ param.alpha) * eye(size(V{1},1));     
        partb = N.^(-1)*Vi*(Y'-Vi'*W0);
        Wi(:,i) = parta\partb;
    end
%     Wi = n2row(Wi);
end

