function [ W0, Wi, D ] = mtmultk( train_x, train_y, param, rank, iternum)

    %% initialization
    Y = train_y';
    V = respchl( train_x, rank );
    W0 = rand(size(train_x{1},2), 1);
    W0 = n2col(W0);
    Wi = rand(size(train_x{1},2), size(train_x{1},1));
    Wi = n2col(Wi);
    D = 0.5*ones(1, size(train_x{1},1));
    D = n2row(D);
    %% Iteration
    IterFolds = iternum;
    
    for iter = 1:IterFolds
%         if(iter == 50)
%             o = 1;   
%         end
        D_old = D;
        
        stall = tic;

        disp(['iteration of training :', num2str(iter)]);
        
        st3 = tic;
        D = Dcal(V, W0, Wi, Y, param, D_old);    
        Dloss = Dobjfun(  V, D, W0, Wi, Y, param ); 
        disp(['loss of D :', num2str(Dloss)]);
        t3 = toc(st3);
        disp(['time of D :', num2str(t3),'s']); 

        st1 = tic;
        W0 = W0cal(V, Wi, D, Y, param);
        W0loss = W0objfun(  V, D, W0, Wi, Y, param );
        disp(['loss of W0 :', num2str(W0loss)]); 
        t1 = toc(st1);
        disp(['time of W0 :', num2str(t1),'s']); 
          
        st2 = tic;
        Wi = Wicalori(V, W0, D, Y, param);
        Wiloss = Wiobjfunori(  V, D, W0, Wi, Y, param );    
        disp(['loss of Wi :', num2str(Wiloss)]); 
        t2 = toc(st2);
        disp(['time of Wi :', num2str(t2),'s']); 
        
        allloss = allobjfun( V, D, W0, Wi, Y, param );
        disp(['loss of this iteration :', num2str(allloss)]);
        
        tall = toc(stall);
        disp(['time of all :', num2str(tall),'s']); 
    end
end

