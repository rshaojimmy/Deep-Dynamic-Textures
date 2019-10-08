function [ test_score ] = test_multk( test_x, rank, W0, Wi, D )
    
    dis = D;
    [redisvect,idx] = sort(dis,'descend');

    idxsel = idx(1:rank);
    
    disN = zeros(1, size(idxsel, 2));
    for chl = 1 : size(idxsel, 2)
        chlnum = idxsel(chl);
        disN(1, chl) = dis(1, chlnum);
%         disN(1, chl) = 1./size(idxsel, 2);
    end
    
    disN = disN./norm(disN);
    
    test_score = zeros(length(test_x), 1);

    for sbjnum = 1 : length(test_x)
        
        sbjval = test_x{sbjnum,1};
        sbjval = sbjval';
        score = zeros(1, size(idxsel, 2));
        for i = 1 : size(idxsel, 2)
            chlidx = idxsel(i);
            if (disN(1, i)>0)
                score(i) = ((W0 + disN(1, i).^(-1)*Wi(:,chlidx))'*sbjval(:,chlidx))*disN(1, i);
            else
                score(i) = 0;
            end
        end
        test_score(sbjnum,:) = sum(score);
    end
%     test_score = test_score./norm(test_score);

end

