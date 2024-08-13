import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pickle
import datetime
from collections import defaultdict


from utils_oop_good_detailed_potting import(wrap_around_and_compress, scaled_single_sigmoid, multiInterp2, generate_spatial_rate_maps, get_exp_decay_filter,
    get_dual_exp_decay_signal_filters, get_target_synthetic_ramp, get_global_signal, calibrate_ramp_scaling_factor,
    validate_matrix, get_ramp_population, get_firing_rates, get_plateau_probability2, get_plateau_times2,
    get_2d_induction_gate, get_two_track_length_ET, update_weights2, Network, plot_network_state, plot_saved_network_state) #plot_network_history


file_path = r'C:\Users\Msfin\cloned_repositories\CRCNS_BTSP\config\simulate_CRCNS_BTSP_1.yaml'
try:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    globals()['config'] = config

    print(config['data_file_name'])
    print(config['input_field_width'])
    print(config['track_length'])
except FileNotFoundError:
    print(f"File not found: {file_path}")




if __name__ == '__main__':
    if config:
        network = Network(config)
        # network.simulate_network(plot=True, export=True) # export=False if exporting after the simulation

# have this uncommented if you want to plot network state after running the whole simulation
#         plot_network_state(network.CA3_input_rates, network.CA1_ramp_pop_history, network.pop_CA1_rate_history, network.SST_ramp_pop_history, network.pop_SST_rate_history, network.CA1_dendrite_vm_history, network.track_phases, network.total_num_laps)

# have this uncommented if you want to load a saved network sim file to work on plotting it
        saved_file_path = r'data\network_simulation_20240813_133228.pkl'

        with open(saved_file_path, 'rb') as file:
            network = pickle.load(file)

        print(f"Loaded network from {saved_file_path}.")
        print(f"Length of SST_ramp_pop_history after loading: {len(network.SST_ramp_pop_history)}")

        plot_saved_network_state(saved_file_path)
#


        # plot_network_history(
        #     saved_file_path,
        #     network.SST_ramp_pop_history,
        #     network.pop_SST_rate_history,
        #     network.track_phases,
        #     network.track_x,
        #     network.basal_plateau_prob_f,
        #     network.reward_plateau_prob_f,
        #     network.dt,
        #     network.basal_representation_xscale,
        #     network.reward_representation_xscale,
        #     network.ramp_xscale,
        #     network.input_field_peak_rate,
        #     network.binned_x,
        #     network.track_length,
        #     network.total_num_laps,
        #     network.f_I)


        # with open(saved_file_path, 'rb') as file:
        #     network = pickle.load(file)
        #
        # print(f"Length of pop_SST_rate_history after loading: {len(network.pop_SST_rate_history)}")
        # print(f"Length of pop_CA1_rate_history after loading: {len(network.pop_CA1_rate_history)}")


    else:
         print("config file could not be loaded")


