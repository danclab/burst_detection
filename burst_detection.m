function bursts=extract_bursts(raw_trials, TF, erf, times, search_freqs, band_lims, fooof_thresh, sfreq, varargin)
% raw_trials = trials x time
% TF = trials x freq x time

    defaults = struct('win_size', .2, 'beh_idx', []);
    params = struct(varargin{:});
    for f = fieldnames(defaults)',
        if ~isfield(params, f{1}),
            params.(f{1}) = defaults.(f{1});
        end
    end

    bursts=[];
    bursts.trial=[];
    bursts.waveform=[];
    bursts.peak_freq=[];
    bursts.peak_amp_iter=[];
    bursts.peak_amp_base=[];
    bursts.peak_time=[];
    bursts.peak_adjustment=[];
    bursts.fwhm_freq=[];
    bursts.fwhm_time=[];
    bursts.polarity=[];

    % Compute erf
    erf = mean(raw_trials);

    % Grid for computing 2D Gaussians
    [x_idx, y_idx] = meshgrid([1:length(times)], [1:len(search_freqs)]);

    % Window size in points
    wlen = round(params.w_size * sfreq);
    half_wlen = round(wlen * .5)

    % Iterate through trials
    for t_idx=1:size(TF,1)
        tr=squeeze(TF(t_idx,:,:));

        % Subtract 1/f threshold
        trial_TF = tr - fooof_thresh;
        trial_TF(trial_TF < 0) = 0;

        % TF for iterating
        trial_TF_iter = trial_TF;

        % Regress out ERF
        lm=fitlm(erf, raw_trials(t_idx,:));
        raw_trials(t_idx,:)=lm.Residuals.Raw;

        while true
            % Compute noise floor
            thresh = 2 * std(trial_TF_iter);

            % Find peak
            [M,I] = max(trial_TF_iter,[],"all","linear");
            [peak_freq_idx, peak_time_idx] = ind2sub(size(trial_TF_iter),I);
            peak_freq = search_freqs(peak_freq_idx);
            peak_amp_iter = trial_TF_iter(peak_freq_idx, peak_time_idx);
            peak_amp_base = trial_TF(peak_freq_idx, peak_time_idx);
            if peak_amp_iter < thresh
                break
            end

            % Fit 2D Gaussian and subtract from TF
            [right_loc, left_loc, up_loc, down_loc] = fwhm_burst_norm(trial_TF_iter, [peak_freq_idx, peak_time_idx]);

            # REMOVE DEGENERATE GAUSSIAN
            vert_isnan = any(isnan([up_loc, down_loc]));
            horiz_isnan = any(isnan([right_loc, left_loc]));
            if vert_isnan
                v_sh = round((length(search_freqs) - peak_freq_idx) / 2);
                if v_sh <= 0
                    v_sh = 1;
                end
                up_loc = v_sh;
                down_loc = v_sh;

            elif horiz_isnan:
                h_sh = round((length(times) - peak_time_idx) / 2);
                if h_sh <= 0
                    h_sh = 1;
                end
                right_loc = h_sh;
                left_loc = h_sh;
            end

            hv_isnan = any([vert_isnan, horiz_isnan]);

            fwhm_f_idx = up_loc + down_loc;
            fwhm_f = (search_freqs(2)-search_freqs(1))*fwhm_f_idx;
            fwhm_t_idx = left_loc + right_loc;
            fwhm_t = (times(2) - times(1))*fwhm_t_idx;
            sigma_t = (fwhm_t_idx) / 2.355;
            sigma_f = (fwhm_f_idx) / 2.355;
            z = peak_amp_iter * gaus2d(x_idx, y_idx, peak_time_idx, peak_freq_idx, sigma_t, sigma_f);
            new_trial_TF_iter = trial_TF_iter - z;

            if peak_freq>=band_lims(1) && peak_freq<=band_lims(2) && ~hv_isnan
                % Extract raw burst signal
                dur = [max([1, peak_time_idx - left_loc]),...
                    min([size(raw_trials,2), peak_time_idx + right_loc])];
                raw_signal = raw_trials(t_idx, dur(1):dur(2));

                % Bandpass filter
                freq_range = [max([1, peak_freq_idx - down_loc]),...
                    min([length(search_freqs) , peak_freq_idx + up_loc])];
                filtered = filter_data(raw_signal, sfreq, search_freqs(freq_range(1)),...
                    search_freqs(freq_range(2)));

                % Hilbert transform
                analytic_signal = hilbert(filtered);
                % Get phase
                instantaneous_phase = mod(np.unwrap(np.angle(analytic_signal)), math.pi);

                % Find phase local minima (near 0)
                zero_phase_pts = argrelextrema(instantaneous_phase', np.less);
                % Find local phase minima with negative deflection closest to TF peak
                closest_pt = zero_phase_pts(argmin(np.abs((dur(2) - dur(1)) * .5 - zero_phase_pts)));
                new_peak_time_idx = dur(1) + closest_pt;
                adjustment = (new_peak_time_idx - peak_time_idx) * 1 / sfreq;

                % Keep if adjustment less than 30ms
                if abs(adjustment) < .03

                    % If burst won't be cutoff
                    if new_peak_time_idx >= half_wlen && new_peak_time_idx + half_wlen <= size(raw_trials,2)
                        peak_time = times(new_peak_time_idx);

                        overlapped=false;
                        % Check for overlap
                        for b_idx=1:length(bursts.peak_time)
                            if bursts.trial(b_idx)==t_idx
                                o_t=bursts.peak_time(b_idx);
                                o_fwhm_t=bursts.fwhm_time(b_idx);
                                if overlap([peak_time-.5*fwhm_t, peak_time+.5*fwhm_t], [o_t-.5*o_fwhm_t, o_t+.5*o_fwhm_t])
                                    overlapped=true;
                                    break
                                end
                            end
                        end

                        if ~overlapped
                            % Get burst
                            burst = raw_trials(t_idx, new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen);
                            % Remove DC offset
                            burst = burst - mean(burst);
                            burst_times = times(new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen) - times(new_peak_time_idx);

                            % Flip if positive deflection
                            peak_dists = abs(argrelextrema(filtered.T, np.greater) - closest_pt);
                            trough_dists = abs(argrelextrema(filtered.T, np.less) - closest_pt);

                            polarity=0;
                            if length(trough_dists) == 0 || (length(peak_dists) > 0 && min(peak_dists) < min(trough_dists))
                                burst = burst*-1.0;
                                polarity=1;
                            end

                            if length(params.beh_idx)>0
                                bursts.trial(end+1)=beh_idx(t_idx);
                            else
                                bursts.trial(end+1)=t_idx;
                            end
                            bursts.waveform(end+1,:)=burst;
                            bursts.peak_freq(end+1)=peak_freq;
                            bursts.peak_amp_iter(end+1)=peak_amp_iter;
                            bursts.peak_amp_base(end+1)=peak_amp_base;
                            bursts.peak_time(end+1)=peak_time;
                            bursts.peak_adjustment(end+1)=adjustment;
                            bursts.fwhm_freq(end+1)=fwhm_f;
                            bursts.fwhm_time(end+1)=fwhm_t;
                            bursts.polarity(end+1)=polarity;
                        end
                    end
                end
            end

            trial_TF_iter = new_trial_TF_iter;
        end
    end
    bursts.waveform_times = burst_times;
end

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
    cand=find(TF(peak(1),1:peak(2)+1)<=half);
    if length(cand)>0
        left_loc = peak(1)-cand(end);
    end

    down_loc = NaN;
    cand=find(TF(1:peak(1)+1,peak(2))<=half);
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

function z=gaus2d(x, y, mx, my, sx, sy)
    z=exp(-((x - mx)^2. / (2. * sx^2.) + (y - my)^2. / (2. * sy^2.)));
end


function o=overlap(a,b)
    o=(a(1)<=b(1) && b(1)<=a(2)) || (b(1)<=a(1) && a(1)<=b(2));
end