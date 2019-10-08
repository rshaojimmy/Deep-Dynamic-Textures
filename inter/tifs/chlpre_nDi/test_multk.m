function [ test_score ] = test_multk( test_x, W0, Wi  )
   
    test_score = zeros(length(test_x), 1);

    for sbjnum = 1 : length(test_x)
        
        sbjval = test_x{sbjnum,1};
        sbjval = sbjval';
        score = zeros(1, size(sbjval, 2));
        for i = 1 : size(sbjval, 2)
            Wione = Wi(:,i);
            score(i) = (W0 + Wione)'*sbjval(:,i);
        end
        test_score(sbjnum,:) = sum(score);
    end
%     test_score = test_score./norm(test_score);

end

