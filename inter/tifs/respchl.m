function [ V ] = respchl( train_x, rank )
    V = {};
    for chlnum = 1 : rank
        for sbjnum = 1 : length(train_x)
            sbjval = train_x{sbjnum};
            chlvalall(sbjnum,:) = sbjval(chlnum,:);
        end
        V{chlnum} = chlvalall';
    end

end

