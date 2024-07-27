
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
num_laps = 25


position = np.append(np.add(default_interp_x, -track_length), default_interp_x)

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


reward_start_times = context.reward_start_times ############# where

lap_start_times = [-len(default_interp_t) * dt, 0.]
for lap in range(num_laps):
    lap_start_times.append(running_t)

# plateau_prob_ramp_sensitivity_f = scaled_single_sigmoid(0., 4., ramp_xscale)
plateau_prob_ramp_sensitivity_f = lambda x: x / 10.
plateau_prob_ramp_sensitivity_f = np.vectorize(plateau_prob_ramp_sensitivity_f)

signal_xrange = np.linspace(0., 1., 10000)




pot_rate = scaled_single_sigmoid(context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange) ########## find  ########## find
dep_rate = scaled_single_sigmoid(context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange)

local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = get_dual_exp_decay_signal_filters(local_signal_decay, global_signal_decay, dt)

####fix might overwrite out current global_filter ###global_filter_t, global_filter = get_exp_rise_decay_filter(global_signal_rise, global_signal_decay, max_time_scale, dt)
down_plateau_len = int(plateau_dur / down_dt)
example_gate_len = max(down_plateau_len, 2 * len(global_filter_t))
example_induction_gate = np.zeros(example_gate_len)
example_induction_gate[:down_plateau_len] = 1.
example_global_signal = get_global_signal(example_induction_gate, global_filter)
global_signal_peak = np.max(example_global_signal)
peak_delta_weight = context.peak_delta_weight   ########find###

peak_basal_plateau_prob_per_dt = 2.2 * context.peak_basal_plateau_prob_per_sec / 1000. * dt   ########find###
peak_reward_plateau_prob_per_dt = 2.2 * context.peak_reward_plateau_prob_per_sec / 1000. * dt ########find###

basal_representation_xscale = np.linspace(0., context.basal_target_representation_density, 10000) ########find###

########then remove the context dependendencies in the below
basal_plateau_prob_f = \
    scaled_single_sigmoid(context.basal_target_representation_density,
                          context.basal_target_representation_density + context.basal_target_representation_density / 3.,
                          basal_representation_xscale, ylim=[peak_basal_plateau_prob_per_dt, 0.])

reward_representation_xscale = np.linspace(0., context.reward_target_representation_density, 10000)
reward_delta_representation_density = context.reward_target_representation_density - context.basal_target_representation_density
reward_plateau_prob_f = \
    scaled_single_sigmoid(context.reward_target_representation_density,
                          context.reward_target_representation_density + context.reward_delta_representation_density / 2.,
                          reward_representation_xscale,
                          ylim=[peak_reward_plateau_prob_per_dt, 0.])