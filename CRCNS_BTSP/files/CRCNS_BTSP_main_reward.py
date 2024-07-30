import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from CRCNS_BTSP_utils import get_ramp_population, multiInterp2, get_population_representation_density2, get_plateau_probability2, get_plateau_times2, get_2d_induction_gate, update_weights2, wrap_around_and_compress, get_target_synthetic_ramp, generate_spatial_rate_maps, scaled_single_sigmoid, get_global_signal, calibrate_ramp_scaling_factor, get_dual_exp_decay_signal_filters, get_exp_decay_filter



file_path = r'C:\Users\Msfin\cloned_repositories\CRCNS_BTSP\files\simulate_CRCNS_BTSP_1.yaml'
try:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    globals()['config'] = config

    print(config['data_file_name'])
    print(config['input_field_width'])
    print(config['track_length'])
except FileNotFoundError:
    print(f"File not found: {file_path}")


num_inputs = config['num_inputs']
input_field_peak_rate = config['input_field_peak_rate']
input_field_width = config['input_field_width']
track_length = config['track_length']
run_vel = config['run_vel']
num_cells = config['num_cells']
initial_ramp_peak_val = config['initial_ramp_peak_val']
target_asymmetry = config['target_asymmetry']
target_peak_shift = config['target_peak_shift']
target_ramp_width = config['target_ramp_width']
input_field_peak_rate = config['input_field_peak_rate']
initial_ramp_peak_val = config['initial_ramp_peak_val']
spike_threshold = config['spike_threshold']
initial_induction_dur = config['initial_induction_dur']
pause_dur = config['pause_dur']
reward_dur = config['reward_dur']
plateau_dur = config['plateau_dur']
basal_plateau_prob_ramp_sensitivity = config['basal_plateau_prob_ramp_sensitivity']
reward_plateau_prob_ramp_sensitivity = config['reward_plateau_prob_ramp_sensitivity']
k_dep = config['k_dep']
k_pot = config['k_pot']
peak_delta_weight = config['peak_delta_weight']
down_dt = config['down_dt']
dt = config['dt']
ramp_scaling_factor = config['ramp_scaling_factor']
target_peak_shift = config['target_peak_shift']
target_asymmetry = config['target_asymmetry']
target_ramp_width = config['target_ramp_width']
total_num_laps = config['total_num_laps']
local_signal_decay = config['local_signal_decay']
global_signal_decay = config['global_signal_decay']
peak_basal_plateau_prob_per_sec = config['peak_basal_plateau_prob_per_sec']
peak_reward_plateau_prob_per_sec = config['peak_reward_plateau_prob_per_sec']
f_dep_half_width = config['f_dep_half_width']
f_dep_th = config['f_dep_th']
f_pot_half_width = config['f_pot_half_width']
f_pot_th = config['f_pot_th']
peak_delta_weight = config['peak_delta_weight']
basal_target_representation_density = config['basal_target_representation_density']
reward_target_representation_density = config['reward_target_representation_density']
track_phases = config['track_phases']
expected_reward_time = config['expected_reward_time']

track_duration = track_length / run_vel * 1000.
dt = 10.  # ms
dx = run_vel * dt / 1000.
binned_dx = track_length / 100.  # cm
binned_x = np.arange(0., track_length+binned_dx/2., binned_dx)[:100] + binned_dx/2.
generic_dx = binned_dx / 100.  # cm
generic_x = np.arange(0., track_length, generic_dx)
generic_position_dt = generic_dx / run_vel * 1000.  #### used to be default_run_vel which was 25 from context
generic_t = np.arange(0., len(generic_x)*generic_position_dt, generic_position_dt)[:len(generic_x)]
default_interp_t = np.arange(0., generic_t[-1], dt)
default_interp_x = np.interp(default_interp_t, generic_t, generic_x)
t = np.append(np.add(default_interp_t, -len(default_interp_t) * dt), default_interp_t)
running_t = len(default_interp_t) * dt
running_position = track_length
track_t = np.arange(0., track_duration, dt)
track_x = np.arange(0., track_length, dx)


position = np.append(np.add(track_x, -track_length), track_x)

target_initial_induction_loc = -target_peak_shift

target_initial_ramp = get_target_synthetic_ramp(
    target_initial_induction_loc,
    ramp_x=binned_x,
    track_length=track_length,
    target_peak_val=initial_ramp_peak_val * 1.4,
    target_min_val=0.,
    target_asymmetry=target_asymmetry,
    target_peak_shift=target_peak_shift,
    target_ramp_width=target_ramp_width
)

f_I_slope = input_field_peak_rate / (initial_ramp_peak_val * 1.4 - spike_threshold)
f_I = np.vectorize(lambda x: 0. if x < spike_threshold else (x - spike_threshold) * f_I_slope) ### vectorue
max_population_rate_sum = np.mean(f_I(target_initial_ramp)) * num_cells




lap_start_times = [-len(track_t) * dt, 0.]
for lap in range(total_num_laps):
    lap_start_times.append(running_t)

plateau_prob_ramp_sensitivity_f = lambda x: x / 10.
plateau_prob_ramp_sensitivity_f = np.vectorize(plateau_prob_ramp_sensitivity_f)

signal_xrange = np.linspace(0., 1., 10000)


pot_rate = scaled_single_sigmoid(f_pot_th, f_pot_th + f_pot_half_width, signal_xrange)
dep_rate = scaled_single_sigmoid(f_dep_th, f_dep_th + f_dep_half_width, signal_xrange)

local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = get_dual_exp_decay_signal_filters(local_signal_decay, global_signal_decay, dt)


down_plateau_len = int(plateau_dur / down_dt)
example_gate_len = max(down_plateau_len, 2 * len(global_filter_t))
example_induction_gate = np.zeros(example_gate_len)
example_induction_gate[:down_plateau_len] = 1.
example_global_signal = get_global_signal(example_induction_gate, global_filter)
global_signal_peak = np.max(example_global_signal)
peak_delta_weight = peak_delta_weight
peak_basal_plateau_prob_per_dt = 2.9 * peak_basal_plateau_prob_per_sec / 1000. * dt
peak_reward_plateau_prob_per_dt = 2.9 * peak_reward_plateau_prob_per_sec / 1000. * dt

basal_representation_xscale = np.linspace(0., basal_target_representation_density, 10000)

basal_plateau_prob_f = \
    scaled_single_sigmoid(basal_target_representation_density,
                          basal_target_representation_density + basal_target_representation_density / 3.,
                          basal_representation_xscale, ylim=[peak_basal_plateau_prob_per_dt, 0.])

reward_representation_xscale = np.linspace(0., reward_target_representation_density, 10000)
reward_delta_representation_density = reward_target_representation_density - basal_target_representation_density
reward_plateau_prob_f = \
    scaled_single_sigmoid(reward_target_representation_density,
                          reward_target_representation_density + reward_delta_representation_density / 2.,
                          reward_representation_xscale,
                          ylim=[peak_reward_plateau_prob_per_dt, 0.])

input_rate_maps, peak_locs = generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width,
                                                        track_length)

CA3_input_rates = np.array(input_rate_maps)

CA3_input_rates2 = multiInterp2(track_x, binned_x, CA3_input_rates)

CA3_input_rates2_3tracks = np.concatenate((CA3_input_rates2, CA3_input_rates2, CA3_input_rates2), axis=1)

two_d_local_signal_filter = local_signal_filter.reshape(1, -1)


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
    print(f"two_track_length_ET.shape after slicing: {two_track_length_ET.shape}")

    return two_track_length_ET


two_track_length_ET = get_two_track_length_ET(CA3_input_rates2_3tracks, two_d_local_signal_filter, track_t)

print(f"CA3_input_rate2 shape {CA3_input_rates2.shape}")
print(f"CA3_input_rates2_3tracks.shape {CA3_input_rates2_3tracks.shape}")
print(f"two_track_length_ET shape: {two_track_length_ET.shape}")

start_time = time.time()

track_duration_ms = len(track_t) * dt
print(f"track_duration_ms {track_duration_ms}")

weights_pop_history = []
ramp_pop_history = []
pop_rep_density_history = []
prev_plateau_start_times = [[] for _ in range(num_cells)]
plateau_start_times_history = [prev_plateau_start_times]

current_phase_index = 0
current_phase = track_phases[current_phase_index]
total_num_laps = sum(phase['num_laps'] for phase in track_phases)
reward_loc = None
reward_start_times = []
lap_start_times = []

initial_weights_population = [np.ones_like(peak_locs) for _ in range(num_cells)]
initial_weights_matrix = np.array(initial_weights_population)
current_weights_population = initial_weights_matrix.copy()
weights_pop_history = []
ramp_pop_history = []
pop_rep_density_history = []
prev_plateau_start_times = [[] for _ in range(num_cells)]
plateau_start_times_history = [prev_plateau_start_times]

for lap in range(total_num_laps):
    if lap >= sum(phase['num_laps'] for phase in track_phases[:current_phase_index + 1]):
        current_phase_index += 1
        if current_phase_index < len(track_phases):
            current_phase = track_phases[current_phase_index]
            print(f"Transitioned to phase: {current_phase['label']}")
        else:
            print("All phases completed.")
            break

    if current_phase['lap_type'] == 'reward':
        reward_loc = current_phase.get('reward_loc', None)
        print(f"Set reward_loc: {reward_loc} cm")
        lap_start_time = lap * track_duration_ms
        reward_start_time = lap_start_time + (reward_loc / track_length) * track_duration_ms
        reward_start_times.append(reward_start_time)

        relative_reward_time = (reward_start_time - lap_start_time) % track_duration_ms
        print(f"Lap {lap + 1}: Reward Time in ms: {reward_start_time} (Relative Position: {relative_reward_time} ms)")

        if abs(relative_reward_time - expected_reward_time) <= dt:
            print(f"Reward occurs correctly at {expected_reward_time} ms within the lap.")
        else:
            print(
                f"Warning: Reward does not occur at {expected_reward_time} ms! Occurs at {relative_reward_time} ms instead.")
    else:
        reward_loc = None
        reward_start_times.append(None)

    lap_start_times.append(lap * track_duration_ms)
    print(f"Lap {lap + 1} Start Time: {lap_start_times[-1]} ms")

    print(f"reward_loc {reward_loc}")
    print(f"current_phase {current_phase}")

    weights_pop_history.append(current_weights_population)

    current_ramp_population = get_ramp_population(current_weights_population, CA3_input_rates,
                                                  ramp_scaling_factor=ramp_scaling_factor)
    multiInterp_ramp_population = multiInterp2(track_x, binned_x, current_ramp_population)
    print(multiInterp_ramp_population.shape)
    ramp_pop_history.append(multiInterp_ramp_population)

    current_pop_representation_density = get_population_representation_density2(multiInterp_ramp_population,
                                                                                max_population_rate_sum, f_I)
    pop_rep_density_history.append(current_pop_representation_density)

    pop_plateau_probability = get_plateau_probability2(multiInterp_ramp_population, current_pop_representation_density,
                                                       prev_plateau_start_times, lap, plateau_dur, pause_dur,
                                                       reward_dur,
                                                       basal_plateau_prob_ramp_sensitivity,
                                                       reward_plateau_prob_ramp_sensitivity, reward_start_times,
                                                       reward_plateau_prob_f, basal_plateau_prob_f, lap_start_times,
                                                       track_t, plateau_prob_ramp_sensitivity_f)

    plateau_start_times, success_matrix = get_plateau_times2(track_t, dt, pop_plateau_probability, 0, lap,
                                                             lap_start_times, plateau_dur, pause_dur)
    plateau_start_times_history.append(plateau_start_times)

    double_track_length_induction_gates, double_length_t = get_2d_induction_gate(plateau_start_times, track_t,
                                                                                 int(plateau_dur / dt),
                                                                                 lap_start_times[lap], dt)
    global_signals_matrix = get_two_track_length_ET(np.concatenate(
        (double_track_length_induction_gates, double_track_length_induction_gates, double_track_length_induction_gates),
        axis=1), global_filter.reshape(1, -1), track_t)
    print(f"global_signals_matrix shape: {global_signals_matrix.shape}")

    current_weights_population = update_weights2(two_track_length_ET,
                                                 global_signals_matrix[:, :two_track_length_ET.shape[1]],
                                                 current_weights_population, pot_rate, dep_rate, k_pot, k_dep, dt,
                                                 peak_delta_weight)
    print(f"current_weights_population after update:\n{current_weights_population}")

    plt.figure()
    plt.imshow(current_weights_population, aspect='auto')
    plt.title('current_weights_population after update')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(pop_plateau_probability, aspect='auto')
    plt.title('population_plateau_probability')
    plt.colorbar()
    plt.show()

    prev_plateau_start_times = plateau_start_times

end_time = time.time()
print(f"time to completion: {end_time - start_time}")

plt.figure()
for i in range(len(pop_rep_density_history)):
    plt.plot(track_x, pop_rep_density_history[i])
plt.title('Population Representation Density History')
plt.xlabel('Position (cm)')
plt.ylabel('Density')
plt.show()

argmax_indexes = np.argmax(multiInterp_ramp_population, axis=1)
sorted_indexes = np.argsort(argmax_indexes)
plt.figure()
plt.imshow(multiInterp_ramp_population[sorted_indexes, :], aspect='auto')
plt.title('Current Ramp Population')
plt.xlabel('Position')
plt.ylabel('Neuron ID')
plt.colorbar()
plt.show()

