function [ output ] = Wiobjfunnew(  V, D, W0, Wiall, Y, param )

    N = length(Y);
    for i = 1 : length(V)
        Vi = V{i};
        Di = D(i);
        Wi = Wiall(:,i);      
        
        partaval = ((Di).^(0.5)*W0 + Di.^(-0.5)*Wi)'*Vi - (Di).^(0.5)*Y;
        parta = N.^(-1)* norm(partaval,'fro').^2;
      
        partb = param.beta*norm(Di.^(-1)*Wi).^2 + param.alpha*norm(Wi).^2;
        
        loss(i) = parta + partb;

    end
    
    output = sum(loss);
end

