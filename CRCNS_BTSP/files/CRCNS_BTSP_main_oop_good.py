import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from CRCNS_BTSP_utils_oop_good import wrap_around_and_compress, scaled_single_sigmoid, multiInterp2, generate_spatial_rate_maps, get_exp_decay_filter, get_dual_exp_decay_signal_filters, get_target_synthetic_ramp, get_global_signal, calibrate_ramp_scaling_factor, validate_matrix, get_ramp_population, get_population_representation_density2, get_plateau_probability2, get_plateau_times2, get_2d_induction_gate, get_two_track_length_ET, update_weights2


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


class Network:
    def __init__(self, config):
        self.num_inputs = config['num_inputs']
        self.input_field_peak_rate = config['input_field_peak_rate']
        self.input_field_width = config['input_field_width']
        self.track_length = config['track_length']
        self.run_vel = config['run_vel']
        self.num_cells = config['num_cells']
        self.initial_ramp_peak_val = config['initial_ramp_peak_val']
        self.target_asymmetry = config['target_asymmetry']
        self.target_peak_shift = config['target_peak_shift']
        self.target_ramp_width = config['target_ramp_width']
        self.input_field_peak_rate = config['input_field_peak_rate']
        self.initial_ramp_peak_val = config['initial_ramp_peak_val']
        self.spike_threshold = config['spike_threshold']
        self.initial_induction_dur = config['initial_induction_dur']
        self.pause_dur = config['pause_dur']
        self.reward_dur = config['reward_dur']
        self.plateau_dur = config['plateau_dur']
        self.basal_plateau_prob_ramp_sensitivity = config['basal_plateau_prob_ramp_sensitivity']
        self.reward_plateau_prob_ramp_sensitivity = config['reward_plateau_prob_ramp_sensitivity']
        self.k_dep = config['k_dep']
        self.k_pot = config['k_pot']
        self.peak_delta_weight = config['peak_delta_weight']
        self.down_dt = config['down_dt']
        self.dt = config['dt']
        self.ramp_scaling_factor = config['ramp_scaling_factor']
        self.target_peak_shift = config['target_peak_shift']
        self.target_asymmetry = config['target_asymmetry']
        self.target_ramp_width = config['target_ramp_width']
        self.total_num_laps = config['total_num_laps']
        self.local_signal_decay = config['local_signal_decay']
        self.global_signal_decay = config['global_signal_decay']
        self.peak_basal_plateau_prob_per_sec = config['peak_basal_plateau_prob_per_sec']
        self.peak_reward_plateau_prob_per_sec = config['peak_reward_plateau_prob_per_sec']
        self.f_dep_half_width = config['f_dep_half_width']
        self.f_dep_th = config['f_dep_th']
        self.f_pot_half_width = config['f_pot_half_width']
        self.f_pot_th = config['f_pot_th']
        self.peak_delta_weight = config['peak_delta_weight']
        self.basal_target_representation_density = config['basal_target_representation_density']
        self.reward_target_representation_density = config['reward_target_representation_density']
        self.track_phases = config['track_phases']
        self.expected_reward_time = config['expected_reward_time']

        self.track_duration = self.track_length / self.run_vel * 1000.
        self.dt = 10.  # ms
        self.dx = self.run_vel * self.dt / 1000.
        self.binned_dx = self.track_length / 100.  # cm
        self.binned_x = np.arange(0., self.track_length + self.binned_dx / 2., self.binned_dx)[:100] + self.binned_dx / 2.
        self.generic_dx = self.binned_dx / 100.  # cm
        self.generic_x = np.arange(0., self.track_length, self.generic_dx)
        self.generic_position_dt = self.generic_dx / self.run_vel * 1000.  #### used to be default_run_vel which was 25 from context
        self.generic_t = np.arange(0., len(self.generic_x) * self.generic_position_dt, self.generic_position_dt)[:len(self.generic_x)]
        self.default_interp_t = np.arange(0., self.generic_t[-1], self.dt)
        self.default_interp_x = np.interp(self.default_interp_t, self.generic_t, self.generic_x)
        self.t = np.append(np.add(self.default_interp_t, -len(self.default_interp_t) * self.dt), self.default_interp_t)
        self.running_t = len(self.default_interp_t) * self.dt
        self.running_position = self.track_length
        self.track_t = np.arange(0., self.track_duration, self.dt)
        self.track_x = np.arange(0., self.track_length, self.dx)
        self.ramp_x = self.binned_x
        self.target_peak_val = self.initial_ramp_peak_val * 1.4
        self.target_min_val = 0.

        self.position = np.append(np.add(self.track_x, -self.track_length), self.track_x)

        self.target_initial_induction_loc = -self.target_peak_shift

        self.target_initial_ramp = get_target_synthetic_ramp(self.target_initial_induction_loc, self.ramp_x, self.track_length, self.target_peak_val, self.target_min_val, self.target_asymmetry, self.target_peak_shift, self.target_ramp_width)

        self.f_I_slope = self.input_field_peak_rate / (self.initial_ramp_peak_val * 1.4 - self.spike_threshold)
        self.f_I = np.vectorize(lambda x: 0. if x < self.spike_threshold else (x - self.spike_threshold) * self.f_I_slope)  ### vectorue
        self.max_population_rate_sum = np.mean(self.f_I(self.target_initial_ramp)) * self.num_cells

        self.lap_start_times = [-len(self.track_t) * self.dt, 0.]
        for lap in range(self.total_num_laps):
            self.lap_start_times.append(self.running_t)

        self.plateau_prob_ramp_sensitivity_f = lambda x: x / 10.
        self.plateau_prob_ramp_sensitivity_f = np.vectorize(self.plateau_prob_ramp_sensitivity_f)

        self.signal_xrange = np.linspace(0., 1., 10000)

        self.pot_rate = scaled_single_sigmoid(self.f_pot_th, self.f_pot_th + self.f_pot_half_width, self.signal_xrange)
        self.dep_rate = scaled_single_sigmoid(self.f_dep_th, self.f_dep_th + self.f_dep_half_width, self.signal_xrange)

        self.local_signal_filter_t, self.local_signal_filter, self.global_filter_t, self.global_filter = get_dual_exp_decay_signal_filters(
            self.local_signal_decay, self.global_signal_decay, self.dt)

        self.down_plateau_len = int(self.plateau_dur / self.down_dt)
        self.example_gate_len = max(self.down_plateau_len, 2 * len(self.global_filter_t))
        self.example_induction_gate = np.zeros(self.example_gate_len)
        self.example_induction_gate[:self.down_plateau_len] = 1.
        self.example_global_signal = get_global_signal(self.example_induction_gate, self.global_filter)
        self.global_signal_peak = np.max(self.example_global_signal)
        self.peak_delta_weight = self.peak_delta_weight
        self.peak_basal_plateau_prob_per_dt = 2.9 * self.peak_basal_plateau_prob_per_sec / 1000. * self.dt
        self.peak_reward_plateau_prob_per_dt = 2.9 * self.peak_reward_plateau_prob_per_sec / 1000. * self.dt

        self.basal_representation_xscale = np.linspace(0., self.basal_target_representation_density, 10000)

        self.basal_plateau_prob_f = \
            scaled_single_sigmoid(self.basal_target_representation_density,
                                  self.basal_target_representation_density + self.basal_target_representation_density / 3.,
                                  self.basal_representation_xscale, ylim=[self.peak_basal_plateau_prob_per_dt, 0.])

        self.reward_representation_xscale = np.linspace(0., self.reward_target_representation_density, 10000)
        self.reward_delta_representation_density = self.reward_target_representation_density - self.basal_target_representation_density
        self.reward_plateau_prob_f = \
            scaled_single_sigmoid(self.reward_target_representation_density,
                                  self.reward_target_representation_density + self.reward_delta_representation_density / 2.,
                                  self.reward_representation_xscale,
                                  ylim=[self.peak_reward_plateau_prob_per_dt, 0.])

        self.input_rate_maps, self.peak_locs = generate_spatial_rate_maps(self.binned_x, self.num_inputs,
                                                                               self.input_field_peak_rate,
                                                                               self.input_field_width,
                                                                               self.track_length)

        self.CA3_input_rates = np.array(self.input_rate_maps)

        self.CA3_input_rates2 = multiInterp2(self.track_x, self.binned_x, self.CA3_input_rates)

        self.CA3_input_rates2_3tracks = np.concatenate(
            (self.CA3_input_rates2, self.CA3_input_rates2, self.CA3_input_rates2), axis=1)

        self.two_d_local_signal_filter = self.local_signal_filter.reshape(1, -1)

        self.two_track_length_ET = get_two_track_length_ET(self.CA3_input_rates2_3tracks, self.two_d_local_signal_filter,
                                                                self.track_t)

    def simulate_network(self):
        start_time = time.time()

        track_duration_ms = len(self.track_t) * self.dt
        print(f"track_duration_ms {track_duration_ms}")

        weights_pop_history = []
        ramp_pop_history = []
        pop_rep_density_history = []
        prev_plateau_start_times = [[] for _ in range(self.num_cells)]
        plateau_start_times_history = [prev_plateau_start_times]

        current_phase_index = 0
        current_phase = self.track_phases[current_phase_index]
        total_num_laps = sum(phase['num_laps'] for phase in self.track_phases)
        reward_loc = None
        reward_start_times = []
        lap_start_times = []

        initial_weights_population = [np.ones_like(self.peak_locs) for _ in range(self.num_cells)]
        initial_weights_matrix = np.array(initial_weights_population)
        current_weights_population = initial_weights_matrix.copy()
        weights_pop_history = []
        ramp_pop_history = []
        pop_rep_density_history = []
        prev_plateau_start_times = [[] for _ in range(self.num_cells)]
        plateau_start_times_history = [prev_plateau_start_times]

        for lap in range(self.total_num_laps):
            if lap >= sum(phase['num_laps'] for phase in self.track_phases[:current_phase_index + 1]):
                current_phase_index += 1
                if current_phase_index < len(self.track_phases):
                    current_phase = self.track_phases[current_phase_index]
                    print(f"Transitioned to phase: {current_phase['label']}")
                else:
                    print("All phases completed.")
                    break

            if current_phase['lap_type'] == 'reward':
                reward_loc = current_phase.get('reward_loc', None)
                print(f"Set reward_loc: {reward_loc} cm")
                lap_start_time = lap * track_duration_ms
                reward_start_time = lap_start_time + (reward_loc / self.track_length) * track_duration_ms
                reward_start_times.append(reward_start_time)

                relative_reward_time = (reward_start_time - lap_start_time) % track_duration_ms
                print(
                    f"Lap {lap + 1}: Reward Time in ms: {reward_start_time} (Relative Position: {relative_reward_time} ms)")

                if abs(relative_reward_time - self.expected_reward_time) <= self.dt:
                    print(f"Reward occurs correctly at {self.expected_reward_time} ms within the lap.")
                else:
                    print(
                        f"Warning: Reward does not occur at {self.expected_reward_time} ms! Occurs at {relative_reward_time} ms instead.")
            else:
                reward_loc = None
                reward_start_times.append(None)

            lap_start_times.append(lap * track_duration_ms)
            print(f"Lap {lap + 1} Start Time: {lap_start_times[-1]} ms")

            print(f"reward_loc {reward_loc}")
            print(f"current_phase {current_phase}")

            weights_pop_history.append(current_weights_population)

            current_ramp_population = get_ramp_population(current_weights_population, self.CA3_input_rates,
                                                          ramp_scaling_factor=self.ramp_scaling_factor)
            multiInterp_ramp_population = multiInterp2(self.track_x, self.binned_x, current_ramp_population)
            print(multiInterp_ramp_population.shape)
            ramp_pop_history.append(multiInterp_ramp_population)

            current_pop_representation_density = get_population_representation_density2(multiInterp_ramp_population,
                                                                                        self.max_population_rate_sum, self.f_I)
            pop_rep_density_history.append(current_pop_representation_density)

            pop_plateau_probability = get_plateau_probability2(multiInterp_ramp_population,
                                                               current_pop_representation_density,
                                                               prev_plateau_start_times, lap, self.plateau_dur, self.pause_dur,
                                                               self.reward_dur,
                                                               self.basal_plateau_prob_ramp_sensitivity,
                                                               self.reward_plateau_prob_ramp_sensitivity, reward_start_times,
                                                               self.reward_plateau_prob_f, self.basal_plateau_prob_f,
                                                               lap_start_times, self.track_t,
                                                               self.plateau_prob_ramp_sensitivity_f)

            plateau_start_times, success_matrix = get_plateau_times2(self.track_t, self.dt, pop_plateau_probability, 0, lap,
                                                                     lap_start_times, self.plateau_dur, self.pause_dur)
            plateau_start_times_history.append(plateau_start_times)

            double_track_length_induction_gates, double_length_t = get_2d_induction_gate(plateau_start_times, self.track_t,
                                                                                         int(self.plateau_dur / self.dt),
                                                                                         lap_start_times[lap], self.dt)
            global_signals_matrix = get_two_track_length_ET(np.concatenate((double_track_length_induction_gates,
                                                                            double_track_length_induction_gates,
                                                                            double_track_length_induction_gates),
                                                                           axis=1), self.global_filter.reshape(1, -1),
                                                            self.track_t)
            print(f"global_signals_matrix shape: {global_signals_matrix.shape}")

            current_weights_population = update_weights2(self.two_track_length_ET,
                                                         global_signals_matrix[:, :self.two_track_length_ET.shape[1]],
                                                         current_weights_population, self.pot_rate, self.dep_rate, self.k_pot, self.k_dep,
                                                         self.dt, self.peak_delta_weight)
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
            plt.plot(self.track_x, pop_rep_density_history[i])
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


if config:
    network = Network(config)
    network.simulate_network()
else:
    print("config file could not be loaded")






