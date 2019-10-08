function [ Wi ] = Wicalori( V, W0, Y, param )

    Wi = zeros(size(V{1},1), length(V));
    parta = zeros(size(V{1},1),size(V{1},1));
    partb = zeros(size(V{1},1),1); 

    for i = 1 : length(V)

        Vi = V{i};
        ViViT = Vi*Vi';
              
        parta =  ViViT + (param.beta)* eye(size(V{1},1));     
        partb = Vi*(Y'-Vi'*W0);
        Wi(:,i) = parta\partb;
    end
  
end

