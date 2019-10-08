function [ output ] = Doneobjfun( Vi, Di, W0, Wi, Y, param )

    partaval = ((Di).^(0.5)*W0 + Di.^(-0.5)*Wi)'*Vi - (Di).^(0.5)*Y;
    parta = norm(partaval,'fro').^2;

    partbval = (Di).^(-1)*Wi;
    partb = param.beta*norm(partbval).^2;
    
    partc = param.theta*(Di).^2;
    
    output = parta + partb + partc;
end

