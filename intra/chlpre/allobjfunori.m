function [ output ] = allobjfunori( V, D, W0, Wi, Y, param )

    for i = 1 : length(V)
        partaval = ((D(i)).^(0.5)*W0 + D(i).^(-0.5)*Wi(:,i))'*V{i} - (D(i)).^(0.5)*Y;
        parta(i) = norm(partaval,'fro').^2 + param.beta*norm((D(i)).^(-1)*Wi(:,i)).^2;
    end
    suma = sum(parta);
    
    sumb = param.nata*norm(W0).^2;
    
    sumc = param.theta*norm(D).^2;
    
    output = suma + sumb + sumc;
end

