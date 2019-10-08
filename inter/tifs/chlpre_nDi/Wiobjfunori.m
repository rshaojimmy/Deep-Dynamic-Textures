function [ output ] = Wiobjfunori(  V, W0, Wiall, Y, param )

    for i = 1 : length(V)
        Vi = V{i};

        Wi = Wiall(:,i);      
        
        partaval = ( W0 +  Wi)'*Vi -  Y;
        parta = norm(partaval,'fro').^2;
      
        partb = param.beta*norm( Wi).^2;
        
        loss(i) = parta + partb;

    end
    
    output = sum(loss);
end

