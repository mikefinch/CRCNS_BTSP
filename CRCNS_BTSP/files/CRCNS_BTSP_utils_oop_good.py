import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def get_target_synthetic_ramp(induction_loc, ramp_x, track_length, target_peak_val=8., target_min_val=0.,
                              target_asymmetry=1.8, target_peak_shift=-10., target_ramp_width=187., plot=False):
    peak_loc = induction_loc + target_peak_shift
    amp = target_peak_val - target_min_val
    extended_x = np.concatenate([ramp_x - track_length, ramp_x, ramp_x + track_length])
    left_width = target_asymmetry / (1. + target_asymmetry) * target_ramp_width + target_peak_shift
    right_width = 1. / (1. + target_asymmetry) * target_ramp_width - target_peak_shift
    left_sigma = 2. * left_width / 3. / np.sqrt(2.)
    right_sigma = 2. * right_width / 3. / np.sqrt(2.)
    left_waveform = amp * np.exp(-((extended_x - peak_loc) / left_sigma) ** 2.)
    right_waveform = amp * np.exp(-((extended_x - peak_loc) / right_sigma) ** 2.)
    peak_index = np.argmax(left_waveform)
    waveform = np.array(left_waveform)
    waveform[peak_index + 1:] = right_waveform[peak_index + 1:]
    waveform = wrap_around_and_compress(waveform, ramp_x)

    waveform -= np.min(waveform)
    waveform /= np.max(waveform)
    waveform *= amp
    waveform += target_min_val

    if plot:
        fig, axes = plt.subplots()
        axes.plot(ramp_x, waveform)
        axes.set_ylabel('Ramp amplitude (mV)')
        axes.set_xlabel('Location (cm)')
        clean_axes(axes)
        fig.show()

    return waveform
def wrap_around_and_compress(waveform, interp_x):
    before = np.array(waveform[:len(interp_x)])
    after = np.array(waveform[2 * len(interp_x):])
    within = np.array(waveform[len(interp_x):2 * len(interp_x)])
    compressed_waveform = within[:len(interp_x)] + before[:len(interp_x)] + after[:len(interp_x)]
    return compressed_waveform

def scaled_single_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError(
            'scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return lambda xi: (target_amp / amp) * (1. / (1. + np.exp(-slope * (xi - th))) - start_val) + ylim[0]

def calibrate_ramp_scaling_factor(ramp_x, input_x, interp_x, input_rate_maps, peak_locs, track_length, target_range,
                                  bounds, beta=2., initial_ramp_scaling_factor=0.002956,
                                  calibration_ramp_width=108.,
                                  calibration_ramp_amp=6., calibration_peak_delta_weight=1.5, plot=False,
                                  verbose=1):

    calibration_sigma = calibration_ramp_width / 3. / np.sqrt(2.)
    calibration_peak_loc = track_length / 2.
    target_ramp = calibration_ramp_amp * np.exp(-((ramp_x - calibration_peak_loc) / calibration_sigma) ** 2.)

    induction_start_loc = calibration_peak_loc + 10.
    induction_stop_loc = calibration_peak_loc + 5.

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
        peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(ramp=target_ramp, induction_loc=induction_start_loc, binned_x=ramp_x,
                                interp_x=interp_x, track_length=track_length)

    if len(target_ramp) != len(input_x):
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        interp_target_ramp = np.array(target_ramp)
    input_matrix = np.multiply(input_rate_maps, initial_ramp_scaling_factor)
    [U, s, Vh] = np.linalg.svd(input_matrix)
    V = Vh.T
    D = np.zeros_like(input_matrix)
    D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
    input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
    delta_weights = interp_target_ramp.dot(input_matrix_inv)

    SVD_delta_weights, SVD_ramp_scaling_factor = \
        get_adjusted_delta_weights_and_ramp_scaling_factor(delta_weights, input_rate_maps,
                                                           calibration_peak_delta_weight,
                                                           calibration_ramp_amp)
    if bounds is not None:
        SVD_delta_weights = np.maximum(np.minimum(SVD_delta_weights, bounds[1]), bounds[0])

    input_matrix = np.multiply(input_rate_maps, SVD_ramp_scaling_factor)
    SVD_model_ramp = SVD_delta_weights.dot(input_matrix)

    result = minimize(get_residual_score, SVD_delta_weights,
                      args=(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, SVD_ramp_scaling_factor,
                            induction_start_loc, track_length, target_range, bounds), method='L-BFGS-B',
                      bounds=[bounds] * len(SVD_delta_weights), options={'disp': verbose > 1, 'maxiter': 100})

    LSA_delta_weights = result.x
    delta_weights, ramp_scaling_factor = \
        get_adjusted_delta_weights_and_ramp_scaling_factor(LSA_delta_weights, input_rate_maps,
                                                           calibration_peak_delta_weight,
                                                           calibration_ramp_amp)

    input_matrix = np.multiply(input_rate_maps, ramp_scaling_factor)
    model_ramp = delta_weights.dot(input_matrix)

    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)
    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
        peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(ramp=model_ramp, induction_loc=induction_start_loc, binned_x=ramp_x,
                                interp_x=interp_x,
                                track_length=track_length)

    if verbose > 1:
        print('target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'],
                                                 ratio['target'], start_loc['target'], peak_loc['target'],
                                                 end_loc['target']))
        print(
            'model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
            ', end_loc: %.1f' % (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'],
                                 start_loc['model'], peak_loc['model'], end_loc['model']))
        print('ramp_scaling_factor: %.6f' % ramp_scaling_factor)

    sys.stdout.flush()

    if plot:
        x_start = induction_start_loc
        x_end = induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(ramp_x, target_ramp, label='Target', color='k')
        axes[0].plot(ramp_x, SVD_model_ramp, label='Model (SVD)', color='r')
        axes[0].plot(ramp_x, model_ramp, label='Model (LSA)', color='c')
        axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[0].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Ramp amplitude (mV)')
        axes[0].set_xlim([0., track_length])
        axes[0].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)

        ylim = max(np.max(SVD_delta_weights), np.max(delta_weights)) + 1.
        ymin = min(np.min(SVD_delta_weights), np.min(delta_weights)) + 1.
        axes[1].plot(peak_locs, SVD_delta_weights + 1., c='r', label='Model (SVD)')
        axes[1].plot(peak_locs, delta_weights + 1., c='c', label='Model (LSA)')
        axes[1].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[1].set_xlabel('Location (cm)')
        axes[1].set_ylabel('Candidate synaptic weights (a.u.)')
        axes[1].set_xlim([0., track_length])
        axes[1].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    return ramp_scaling_factor

def generate_spatial_rate_maps(x, n=200, peak_rate=1., field_width=90., track_length=187.):
    """
    Return a list of spatial rate maps with peak locations that span the track. Return firing rate vs. location
    computed at the resolution of the provided x array.
    :param x: array
    :param n: int
    :param peak_rate: float
    :param field_width: float
    :param track_length: float
    :return: list of array, array
    """
    gauss_sigma = field_width / 3. / np.sqrt(2.)  # contains 99.7% gaussian area
    d_peak_locs = track_length / float(n)
    peak_locs = np.arange(d_peak_locs / 2., track_length, d_peak_locs)
    spatial_rate_maps = []
    extended_x = np.concatenate([x - track_length, x, x + track_length])
    for peak_loc in peak_locs:
        gauss_force = peak_rate * np.exp(-((extended_x - peak_loc) / gauss_sigma) ** 2.)
        gauss_force = wrap_around_and_compress(gauss_force, x)
        spatial_rate_maps.append(gauss_force)
    return spatial_rate_maps, peak_locs

def get_exp_decay_filter(decay, max_time_scale, dt):  ##utils

        filter_t = np.arange(0., 6. * max_time_scale, dt)
        filter = np.exp(-filter_t / decay)
        decay_indexes = np.where(filter < 0.001 * np.max(filter))[0]
        if np.any(decay_indexes):
            filter = filter[:decay_indexes[0]]
        filter /= np.sum(filter)
        filter_t = filter_t[:len(filter)]
        return filter_t, filter

def get_dual_exp_decay_signal_filters(local_signal_decay, global_signal_decay, dt, plot=False):
    max_time_scale = max(local_signal_decay, global_signal_decay)
    local_signal_filter_t, local_signal_filter = \
        get_exp_decay_filter(local_signal_decay, max_time_scale, dt)
    global_filter_t, global_filter = \
        get_exp_decay_filter(global_signal_decay, max_time_scale, dt)
    if plot:
        fig, axes = plt.subplots(1)
        axes.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='r',
                  label='Local plasticity signal filter')
        axes.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                  label='Global plasticity signal filter')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Normalized filter amplitude')
        axes.set_title('Plasticity signal filters')
        axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes.set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    return local_signal_filter_t, local_signal_filter, global_filter_t, global_filter

def get_global_signal(induction_gate, global_filter):

    return np.convolve(induction_gate, global_filter)[:len(induction_gate)]

def get_ramp_population(weight_matrix, rate_matrix, baseline_weight=1., ramp_scaling_factor=1.):
    print(f"weight_matrix shape: {weight_matrix.shape}")
    print(f"rate_matrix shape: {rate_matrix.shape}")
    print(f"baseline_weight: {baseline_weight}")
    print(f"ramp_scaling_factor: {ramp_scaling_factor}")

    delta_weights = np.subtract(weight_matrix, baseline_weight)
    print(f"delta_weights:\n{delta_weights}")

    if np.all(delta_weights == 0):
        print("delta_weights are all zeros")

    ramp_matrix = delta_weights @ rate_matrix
    print(f"ramp_matrix before scaling:\n{ramp_matrix}")

    if np.all(ramp_matrix == 0):
        print("ramp_matrix is all zeros before scaling")

    ramp_matrix *= ramp_scaling_factor
    print(f"ramp_matrix after scaling:\n{ramp_matrix}")

    if np.all(ramp_matrix == 0):
        print("ramp_matrix is all zeros after scaling")

    return ramp_matrix

def multiInterp2(x, xp, fp, debug=False):
    if debug:
        print(f"multiInterp2 - x shape: {x.shape}, xp shape: {xp.shape}, fp shape: {fp.shape}")

    if fp.shape[1] != len(xp):
        print(f"multiInterp2 - fp and xp size mismatch: fp shape: {fp.shape}, xp size: {len(xp)}")
        raise ValueError("fp columns must match length of xp for interpolation")

    i = np.arange(x.size)
    j = np.searchsorted(xp, x) - 1
    j = np.clip(j, 0, len(xp) - 2)
    d = (x - xp[j]) / (xp[j + 1] - xp[j])

    if debug:
        print(f"multiInterp2 - Indices j: {j}, distances d: {d}")

    result = (1 - d) * fp[:, j] + fp[:, j + 1] * d

    return result

def validate_matrix(matrix, name):
    if np.any(np.isnan(matrix)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(matrix)):
        raise ValueError(f"{name} contains infinite values")


def get_population_representation_density2(multiInterp_ramp_population, max_population_rate_sum, f_I):
    """

    :param ramp_population: list of array (like binned_x)
    :return: array (like default_interp_t)
    """
    binned_population_rate_sum = np.sum(f_I(multiInterp_ramp_population), axis=0)
    population_representation_density = binned_population_rate_sum / max_population_rate_sum
    return population_representation_density


def get_plateau_probability2(multiInterp_ramp_population, population_representation_density2,
                             prev_plateau_start_times, lap, plateau_dur, pause_dur, reward_dur,
                             basal_plateau_prob_ramp_sensitivity, reward_plateau_prob_ramp_sensitivity,
                             reward_start_times, reward_plateau_prob_f, basal_plateau_prob_f, lap_start_times,
                             track_t, plateau_prob_ramp_sensitivity_f):
    start_time = time.time()

    population_representation_density_row_array = population_representation_density2.reshape(1, -1)
    t_start_time = lap_start_times[lap]
    this_t = track_t + t_start_time
    plateau_prob_ramp_modulation_matrix = plateau_prob_ramp_sensitivity_f(multiInterp_ramp_population)

    plateau_prob_row_array = basal_plateau_prob_f(population_representation_density_row_array)
    print("Initial plateau_prob_row_array:", plateau_prob_row_array)

    plateau_prob_row_array = np.maximum(plateau_prob_row_array, 0)
    print("Corrected plateau_prob_row_array:", plateau_prob_row_array)

    plateau_prob_row_array[plateau_prob_row_array < 0] = 0

    plateau_prob_matrix = plateau_prob_row_array * (
                1. + basal_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation_matrix)
    print("Initial plateau_prob_matrix (before reward):", plateau_prob_matrix)

    if reward_start_times[lap] is not None:
        this_reward_start_time = reward_start_times[lap]
        this_reward_stop_time = this_reward_start_time + reward_dur
        reward_indexes = np.where((this_t >= this_reward_start_time) & (this_t < this_reward_stop_time))[0]
        if reward_indexes.size > 0:
            reward_plateau_prob_row_array = reward_plateau_prob_f(
                population_representation_density_row_array[:, reward_indexes])
            print("reward_plateau_prob_row_array:", reward_plateau_prob_row_array)
            reward_plateau_prob_matrix = reward_plateau_prob_row_array * (
                        1. + reward_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation_matrix[:,
                                                                    reward_indexes])
            plateau_prob_matrix[:, reward_indexes] = reward_plateau_prob_matrix
            print("Updated plateau_prob_matrix (with reward):", plateau_prob_matrix)

    prev_lap = lap - 1

    if prev_lap >= 0 and reward_start_times[prev_lap] is not None:
        reward_start_time_prev_lap = reward_start_times[prev_lap]
        reward_stop_time_prev_lap = reward_start_time_prev_lap + reward_dur
        reward_indexes_prev_lap = \
        np.where((this_t >= reward_start_time_prev_lap) & (this_t < reward_stop_time_prev_lap))[0]
        if reward_stop_time_prev_lap > t_start_time:
            reward_plateau_prob_row_array_prev_lap = reward_plateau_prob_f(
                population_representation_density_row_array[:, reward_indexes_prev_lap])
            reward_plateau_prob_matrix_prev_lap = reward_plateau_prob_row_array_prev_lap * (
                        1. + reward_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation_matrix[:,
                                                                    reward_indexes_prev_lap])
            plateau_prob_matrix[:, reward_indexes_prev_lap] = reward_plateau_prob_matrix_prev_lap
            print("Updated plateau_prob_matrix (previous lap reward):", plateau_prob_matrix)

    for unit_id, this_plateau_start_times_list in enumerate(prev_plateau_start_times):
        for this_plateau_start_time in this_plateau_start_times_list:
            this_plateau_stop_time = this_plateau_start_time + plateau_dur + pause_dur
            plateau_indexes = np.where((this_t >= this_plateau_start_time) & (this_t < this_plateau_stop_time))[0]
            if len(plateau_indexes) > 0:
                plateau_prob_matrix[unit_id, plateau_indexes] = 0.

    print(
        f"before clipping the max and min plateau prob matrix min = {np.min(plateau_prob_matrix)} max = {np.max(plateau_prob_matrix)}")
    plateau_prob_matrix = np.clip(plateau_prob_matrix, 0., 1.)
    print(
        f"after clipping the max and min plateau prob matrix min = {np.min(plateau_prob_matrix)} max = {np.max(plateau_prob_matrix)}")

    validate_matrix(plateau_prob_matrix, 'plateau_prob_matrix')

    plateau_prob_matrix = np.nan_to_num(plateau_prob_matrix, nan=0.0)

    if not np.all((plateau_prob_matrix >= 0.) & (plateau_prob_matrix <= 1.)):
        raise Exception('plateau_prob_matrix contains invalid values')

    end_time = time.time()
    print(f"execution time: {end_time - start_time}")
    print("lap: ", lap)

    return plateau_prob_matrix

def get_plateau_times2(track_t, dt, plateau_probability2, network_seed, lap, lap_start_times, plateau_dur,
                       pause_dur, debug=False):
    """
    :param plateau_probability2: array
    :param lap: int
    :param cell_id: int
    :param dt: float
    :return: list of float
    """
    local_random = np.random.RandomState()
    local_random.seed([int(network_seed), int(lap)])

    t_start_time = lap_start_times[lap]
    this_t = track_t + t_start_time
    plateau_len = int(plateau_dur / dt)
    pause_len = int(pause_dur / dt)

    success_matrix = local_random.binomial(1, plateau_probability2)

    plateau_start_times = []
    for i in range(success_matrix.shape[0]):
        this_unit_plateau_start_times = []
        success_indices = np.where(success_matrix[i])[0]
        if debug:
            print(f"Cell {i} success_indices: {success_indices}")
        if len(success_indices) > 1:
            if debug:
                print('getting here with row i:', i)
            intervals_between_successes = np.diff(success_indices)
            invalid_success_indexes = np.where(intervals_between_successes < (plateau_len + pause_len))[0] + 1
            success_matrix[i, success_indices[invalid_success_indexes]] = 0
            success_indices = np.delete(success_indices, invalid_success_indexes)
        if debug:
            print(f"Cell {i} valid success_indices: {success_indices}")
        this_unit_plateau_start_times.extend(this_t[success_indices])
        plateau_start_times.append(this_unit_plateau_start_times)

    non_empty_success_indices_count = 0
    for success_indices in success_matrix:
        if np.any(success_indices):
            non_empty_success_indices_count += 1

    if debug:
        print(f"Number of non-empty success indices: {non_empty_success_indices_count}")

    return plateau_start_times, success_matrix

def get_2d_induction_gate(plateau_start_times, track_t, plateau_len, t_start_time, dt):
    track_duration = len(track_t) * dt
    double_length_t = np.concatenate((track_t, track_t + track_duration))
    print(f"track_t shape {track_t.shape}")
    print(f"double_length_t {double_length_t.shape}")
    double_length_this_t = double_length_t + t_start_time
    print(f"double_length_this_t {double_length_this_t.shape}")
    num_neurons = len(plateau_start_times)
    induction_gates = np.zeros((num_neurons, len(double_length_this_t)))
    print(f"induction_gates shape {induction_gates.shape}")

    for neuron_idx in range(num_neurons):
        neuron_plateau_start_times = plateau_start_times[neuron_idx]

        for plateau_start_time in neuron_plateau_start_times:
            if plateau_start_time:
                start_index = np.where((double_length_this_t) >= plateau_start_time)[0]
                if start_index.size > 0:
                    start_index = start_index[0]
                    induction_gates[neuron_idx, start_index:start_index + plateau_len] = 1.

    return induction_gates, double_length_t

def update_weights2(two_track_length_ET, double_track_length_IS, current_weights, pot_rate, dep_rate, k_pot, k_dep,
                    dt, peak_delta_weight):
    start_time = time.time()

    ET = two_track_length_ET.reshape(1, two_track_length_ET.shape[0], two_track_length_ET.shape[1])
    print(f"reshaped ET {ET.shape}")

    IS = double_track_length_IS.reshape(double_track_length_IS.shape[0], 1, double_track_length_IS.shape[1])
    print(f"reshaped IS {IS.shape}")

    ETxIS = ET * IS
    print(f"ETxIS shape {ETxIS.shape}")

    pot_rate_values = np.clip(pot_rate(ETxIS), 0., 1.)
    dep_rate_values = np.clip(dep_rate(ETxIS), 0., 1.)

    this_pot_rate = np.trapz(pot_rate_values, axis=2, dx=dt / 1000.)
    this_dep_rate = np.trapz(dep_rate_values, axis=2, dx=dt / 1000.)

    print(f"this_pot_rate shape {this_pot_rate.shape}")
    print(f"this_dep_rate shape {this_dep_rate.shape}")

    peak_weight = peak_delta_weight + 1.
    current_normalized_weights = np.divide(current_weights, peak_weight)

    this_normalized_delta_weight = k_pot * this_pot_rate * (1. - current_normalized_weights) - \
                                   k_dep * this_dep_rate * current_normalized_weights
    print(f"this_normalized_delta_weight shape {this_normalized_delta_weight.shape}")

    plt.figure()
    plt.imshow(this_normalized_delta_weight, aspect='auto')
    plt.title("this_normalized_delta_weight")
    plt.colorbar()
    plt.show()

    next_weights_matrix = np.clip(current_normalized_weights + this_normalized_delta_weight, 0., 1.) * peak_weight

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time {execution_time}")

    return next_weights_matrix

def get_two_track_length_ET(CA3_input_rates2_3tracks, two_d_local_signal_filter, track_t):
    local_signals_matrix = convolve2d(CA3_input_rates2_3tracks, two_d_local_signal_filter, mode='full')[:,
                           :CA3_input_rates2_3tracks.shape[1]]
    local_signals_matrix_max = np.max(local_signals_matrix)
    normalized_local_signals_matrix = local_signals_matrix / local_signals_matrix_max

    local_signals_matrix_max = np.max(local_signals_matrix, axis=1, keepdims=True)
    local_signals_matrix_max[local_signals_matrix_max == 0] = np.nan
    normalized_local_signals_matrix = np.divide(local_signals_matrix, local_signals_matrix_max)
    normalized_local_signals_matrix = np.nan_to_num(normalized_local_signals_matrix, nan=0.0)

    start_ET = len(track_t)
    end_ET = start_ET + len(track_t) * 2
    print(f"start_ET: {start_ET}, end_ET: {end_ET}")

    two_track_length_ET = normalized_local_signals_matrix[:, start_ET:end_ET]


    return two_track_length_ET


