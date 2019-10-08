function [ dvlp_score ] = dvlp_multk( dvlp_x, rank, W0, D )
    
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
    
    dvlp_score = zeros(length(dvlp_x), 1);
    for sbjnum = 1 : length(dvlp_x)
        
        sbjval = dvlp_x{sbjnum,1};
        sbjval = sbjval';
        score = zeros(1, size(idxsel, 2));
        for i = 1 : size(idxsel, 2)
            chlidx = idxsel(i);
            score(i) = (W0'*sbjval(:,chlidx))*disN(1, i);
        end
        dvlp_score(sbjnum,:) = sum(score);
    end
%     dvlp_score = dvlp_score./norm(dvlp_score);

end

