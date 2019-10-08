function [ output ] = Dobjfun(  V, D, W0, Wiall, Y, param )

    for i = 1 : length(V)
        Vi = V{i};
        Yn = Y;
        if any(any(isnan(Vi)))             
            [m,n]=find(isnan(Vi)==1);
            Vi(:,n)=[];
            Yn(n)=[];
        end
        
        Di = D(i);
        Wi = Wiall(:,i);      
        loss(i) = Doneobjfun( Vi, Di, W0, Wi, Yn, param );
    end
    
    output = sum(loss);
end

