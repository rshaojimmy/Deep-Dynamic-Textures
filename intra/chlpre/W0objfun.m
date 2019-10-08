function [ output ] = W0objfun(  V, D, W0, Wiall, Y, param )

    for i = 1 : length(V)
        Vi = V{i};
        Di = D(i);
        Wi = Wiall(:,i);      
        
        partaval = ((Di).^(0.5)*W0 + Di.^(-0.5)*Wi)'*Vi - (Di).^(0.5)*Y;
        parta = norm(partaval,'fro').^2;
              
        lossa(i) = parta;
    end
    
    outputa = sum(lossa);
    outputb = param.nata*norm(W0).^2;
    output = outputa + outputb;
end

