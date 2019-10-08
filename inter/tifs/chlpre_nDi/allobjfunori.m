function [ output ] = allobjfunori( V, W0, Wi, Y, param )

    for i = 1 : length(V)
        partaval = ( W0 +  Wi(:,i))'*V{i} -  Y;
        parta(i) = norm(partaval,'fro').^2 + param.beta*norm( Wi(:,i)).^2;
    end
    suma = sum(parta);
    
    sumb = param.nata*norm(W0).^2;
    
    
    output = suma + sumb;
end

