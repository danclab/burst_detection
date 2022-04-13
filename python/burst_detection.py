import copy
import math

import numpy as np
from mne.filter import filter_data
from scipy.signal import hilbert, argrelextrema
from scipy.stats import linregress


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def overlap(a, b):
    return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]


def fwhm_burst_norm(TF, peak):
    right_loc = np.nan
    cand = np.where(TF[peak[0], peak[1]:] <= TF[peak] / 2)[0]
    if len(cand):
        right_loc = cand[0]

    up_loc = np.nan
    cand = np.where(TF[peak[0]:, peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        up_loc = cand[0]

    left_loc = np.nan
    cand = np.where(TF[peak[0], :peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        left_loc = peak[1] - cand[-1]

    down_loc = np.nan
    cand = np.where(TF[:peak[0], peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        down_loc = peak[0] - cand[-1]

    if down_loc is np.nan:
        down_loc = up_loc
    if up_loc is np.nan:
        up_loc = down_loc
    if left_loc is np.nan:
        left_loc = right_loc
    if right_loc is np.nan:
        right_loc = left_loc

    horiz = np.nanmin([left_loc, right_loc])
    vert = np.nanmin([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert
    return right_loc, left_loc, up_loc, down_loc


def extract_bursts(raw_trials, TF, times, search_freqs, band_lims, fooof_thresh, sfreq, beh_ix=None, w_size=.2):
    bursts = {
        'trial': [],
        'waveform': [],
        'peak_freq': [],
        'peak_amp_iter': [],
        'peak_amp_base': [],
        'peak_time': [],
        'peak_adjustment': [],
        'fwhm_freq': [],
        'fwhm_time': [],
        'polarity': [],
    }

    # Compute ERF
    erf=np.mean(raw_trials, axis=0)

    # Grid for computing 2D Gaussians
    x_idx, y_idx = np.meshgrid(range(len(times)), range(len(search_freqs)))

    # Window size in points
    wlen = int(w_size * sfreq)
    half_wlen = int(wlen * .5)

    # Iterate through trials
    for t_idx, tr in enumerate(TF):

        # Subtract 1/f threshold
        trial_TF = tr - fooof_thresh
        trial_TF[trial_TF < 0] = 0

        # TF for iterating
        trial_TF_iter = copy.copy(trial_TF)

        # Regress out ERF
        slope, intercept, r, p, se = linregress(erf, raw_trials[t_idx, :])
        raw_trials[t_idx, :] = raw_trials[t_idx, :] - (intercept + slope * erf)

        while True:
            # Compute noise floor
            thresh = 2 * np.std(trial_TF_iter)

            # Find peak
            [peak_freq_idx, peak_time_idx] = np.unravel_index(np.argmax(trial_TF_iter), trial_TF.shape)
            peak_freq = search_freqs[peak_freq_idx]
            peak_amp_iter = trial_TF_iter[peak_freq_idx, peak_time_idx]
            peak_amp_base = trial_TF[peak_freq_idx, peak_time_idx]
            if peak_amp_iter < thresh:
                break

            # Fit 2D Gaussian and subtract from TF
            right_loc, left_loc, up_loc, down_loc = fwhm_burst_norm(trial_TF_iter, (peak_freq_idx, peak_time_idx))

            # REMOVE DEGENERATE GAUSSIAN
            vert_isnan = any(np.isnan([up_loc, down_loc]))
            horiz_isnan = any(np.isnan([right_loc, left_loc]))
            if vert_isnan:
                v_sh = int((search_freqs.shape[0] - peak_freq_idx) / 2)
                if v_sh <= 0:
                    v_sh = 1
                up_loc = v_sh
                down_loc = v_sh

            elif horiz_isnan:
                h_sh = int((times.shape[0] - peak_time_idx) / 2)
                if h_sh <= 0:
                    h_sh = 1
                right_loc = h_sh
                left_loc = h_sh

            hv_isnan = any([vert_isnan, horiz_isnan])

            fwhm_f_idx = up_loc + down_loc
            fwhm_f = (search_freqs[1] - search_freqs[0]) * fwhm_f_idx
            fwhm_t_idx = left_loc + right_loc
            fwhm_t = (times[1] - times[0]) * fwhm_t_idx
            sigma_t = (fwhm_t_idx) / 2.355
            sigma_f = (fwhm_f_idx) / 2.355
            z = peak_amp_iter * gaus2d(x_idx, y_idx, mx=peak_time_idx, my=peak_freq_idx, sx=sigma_t, sy=sigma_f)
            new_trial_TF_iter = trial_TF_iter - z

            if all([peak_freq >= band_lims[0], peak_freq <= band_lims[1], not hv_isnan]):
                # Extract raw burst signal
                dur = [
                    np.max([0, peak_time_idx - left_loc]),
                    np.min([raw_trials.shape[1], peak_time_idx + right_loc])
                ]
                raw_signal = raw_trials[t_idx, dur[0]:dur[1]].reshape(1, -1)

                # Bandpass filter
                freq_range = [
                    np.max([0, peak_freq_idx - down_loc]),
                    np.min([len(search_freqs) - 1, peak_freq_idx + up_loc])
                ]
                filtered = filter_data(raw_signal, sfreq, search_freqs[freq_range[0]], search_freqs[freq_range[1]],
                                       verbose=False)

                # Hilbert transform
                analytic_signal = hilbert(filtered)
                # Get phase
                instantaneous_phase = np.unwrap(np.angle(analytic_signal)) % math.pi

                # Find phase local minima (near 0)
                zero_phase_pts = argrelextrema(instantaneous_phase.T, np.less)[0]
                # Find local phase minima with negative deflection closest to TF peak
                closest_pt = zero_phase_pts[np.argmin(np.abs((dur[1] - dur[0]) * .5 - zero_phase_pts))]
                new_peak_time_idx = dur[0] + closest_pt
                adjustment = (new_peak_time_idx - peak_time_idx) * 1 / sfreq

                # Keep if adjustment less than 30ms
                if np.abs(adjustment) < .03:

                    # If burst won't be cutoff
                    if new_peak_time_idx >= half_wlen and new_peak_time_idx + half_wlen <= raw_trials.shape[1]:
                        peak_time = times[new_peak_time_idx]

                        overlapped = False
                        t_bursts = np.where(bursts['trial'] == t_idx)[0]
                        # Check for overlap
                        for b_idx in t_bursts:
                            o_t = bursts['peak_time'][b_idx]
                            o_fwhm_t = bursts['fwhm_time'][b_idx]
                            if overlap([peak_time - .5 * fwhm_t, peak_time + .5 * fwhm_t],
                                       [o_t - .5 * o_fwhm_t, o_t + .5 * o_fwhm_t]):
                                overlapped = True
                                break

                        if not overlapped:
                            # Get burst
                            burst = raw_trials[t_idx, new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen]
                            ## Remove DC offset
                            burst = burst - np.mean(burst)
                            burst_times = times[new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen] - times[
                                new_peak_time_idx]

                            # Flip if positive deflection
                            peak_dists = np.abs(argrelextrema(filtered.T, np.greater)[0] - closest_pt)
                            trough_dists = np.abs(argrelextrema(filtered.T, np.less)[0] - closest_pt)

                            polarity = 0
                            if len(trough_dists) == 0 or (
                                    len(peak_dists) > 0 and np.min(peak_dists) < np.min(trough_dists)):
                                burst *= -1.0
                                polarity = 1

                            if (type(beh_ix) == list) and (len(beh_ix) == len(TF)):
                                bursts['trial'].append(int(beh_ix[t_idx]))
                            else:
                                print("no beh_match list or not equivalent")
                                bursts['trial'].append(int(t_idx))

                            bursts['waveform'].append(burst)
                            bursts['peak_freq'].append(peak_freq)
                            bursts['peak_amp_iter'].append(peak_amp_iter)
                            bursts['peak_amp_base'].append(peak_amp_base)
                            bursts['peak_time'].append(peak_time)
                            bursts['peak_adjustment'].append(adjustment)
                            bursts['fwhm_freq'].append(fwhm_f)
                            bursts['fwhm_time'].append(fwhm_t)
                            bursts['polarity'].append(polarity)

            trial_TF_iter = new_trial_TF_iter

    bursts['trial'] = np.array(bursts['trial'])
    bursts['waveform'] = np.array(bursts['waveform'])
    bursts['waveform_times'] = burst_times
    bursts['peak_freq'] = np.array(bursts['peak_freq'])
    bursts['peak_amp_iter'] = np.array(bursts['peak_amp_iter'])
    bursts['peak_amp_base'] = np.array(bursts['peak_amp_base'])
    bursts['peak_time'] = np.array(bursts['peak_time'])
    bursts['peak_adjustment'] = np.array(bursts['peak_adjustment'])
    bursts['fwhm_freq'] = np.array(bursts['fwhm_freq'])
    bursts['fwhm_time'] = np.array(bursts['fwhm_time'])
    bursts['polarity'] = np.array(bursts['polarity'])

    return bursts