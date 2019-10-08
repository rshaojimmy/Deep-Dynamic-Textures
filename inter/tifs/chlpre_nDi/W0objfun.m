function [ output ] = W0objfun(  V, W0, Wiall, Y, param )


    for i = 1 : length(V)
        Vi = V{i};

        Wi = Wiall(:,i);      
        
        partaval = ( W0 + Wi)'*Vi -  Y;
        parta = norm(partaval,'fro').^2;
              
        lossa(i) = parta;
    end
    
    outputa = sum(lossa);
    outputb = param.nata*norm(W0).^2;
    output = outputa + outputb;
end

