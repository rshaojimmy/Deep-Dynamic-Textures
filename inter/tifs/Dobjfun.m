function [ output ] = Dobjfun(  V, D, W0, Wiall, Y, param )

    for i = 1 : length(V)
        Vi = V{i};
        Di = D(i);
        Wi = Wiall(:,i);      
        loss(i) = Doneobjfun( Vi, Di, W0, Wi, Y, param );
    end
    
    output = sum(loss);
end

