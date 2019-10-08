function [ output ] = Doneobjfunnew( Vi, Di, W0, Wi, Y, param )

    N = length(Y);
    
    partaval = ((Di).^(0.5)*W0 + Di.^(-0.5)*Wi)'*Vi - (Di).^(0.5)*Y;
    parta = N.^(-1)*norm(partaval,'fro').^2;

    partbval = (Di).^(-1)*Wi;
    partb = param.beta*norm(partbval).^2;
    
    partc = param.theta*(Di).^2;
    
    output = parta + partb + partc;
end

