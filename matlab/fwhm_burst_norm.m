function [right_loc, left_loc, up_loc, down_loc]=fwhm_burst_norm(TF, peak)
    half=TF(peak(1),peak(2))/2;
    right_loc = NaN;
    cand=find(TF(peak(1),peak(2):end)<=half);
    if length(cand)>0
        right_loc=cand(1);
    end

    up_loc = NaN;
    cand=find(TF(peak(1):end, peak(2)) <= half);
    if length(cand)>0
        up_loc=cand(1);
    end

    left_loc = NaN;
    cand=find(TF(peak(1),1:peak(2)-1)<=half);
    if length(cand)>0
        left_loc = peak(2)-cand(end);
    end

    down_loc = NaN;
    cand=find(TF(1:peak(1)-1,peak(2))<=half);
    if length(cand)>0
        down_loc = peak(1)-cand(end);
    end

    if isnan(down_loc)
        down_loc = up_loc;
    end
    if isnan(up_loc)
        up_loc = down_loc;
    end
    if isnan(left_loc)
        left_loc = right_loc;
    end
    if isnan(right_loc)
        right_loc = left_loc;
    end

    horiz = min([left_loc, right_loc]);
    vert = min([up_loc, down_loc]);
    right_loc = horiz;
    left_loc = horiz;
    up_loc = vert;
    down_loc = vert;
end