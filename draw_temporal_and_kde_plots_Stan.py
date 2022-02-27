import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# increase the bin size from 0.36s to the 0.36s*multiplier
def increase_bin_size(bin_divide, trace):
    new_trace = trace.reshape([-1, bin_divide, trace.shape[1]]).sum(1)
    return new_trace

def read_actual(vid):
    path_in_actual = os.getcwd()+'/data/Stan/actual/'

    vid_path = path_in_actual + '/' + vid
    df_actual_arr = pd.read_csv(vid_path)[[' addr2_bytes']].values
    bin_divide = 4
    df_actual_arr = increase_bin_size(bin_divide=bin_divide, trace=df_actual_arr)

    return df_actual_arr


def read_synth(vid):
    path_in_synth = os.getcwd()+'/data/Stan/synth/'

    vid_path = path_in_synth + '/' + vid
    features = [' addr2_bytes']
    df_synth_arr = pd.read_csv(vid_path)[features].values[:125, :]
    return df_synth_arr


# kde plot comparison between the very initially generated traces and the new algorithms
def plot_kde_and_time(data_1, data_2, label_1, label_2):

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    sns.kdeplot(data_1, shade=True, color='b', linewidth=2, label=label_1, ax=ax)
    sns.kdeplot(data_2, shade=True, color='orange', linewidth=2, label=label_2, ax=ax)
    ax.legend(fontsize=50)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=50)
    ax.set_ylabel('Density', fontsize=60)
    ax.set_xlabel('Data dl (MB)', fontsize=60)
    ax.grid()
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    t = np.linspace(0, 180, len(data_1))
    ax.plot(t, data_1, color='b', linewidth=2, linestyle='--', label=label_1)
    ax.plot(t, data_2, color='orange', linewidth=2, linestyle='--', label=label_2)
    ax.legend(fontsize=50)
    ax.grid()
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=50)
    ax.set_xlabel('Time (s)', fontsize=60)
    ax.set_ylabel('Data dl (MB)', fontsize=60)
    plt.tight_layout()
    plt.show()



def draw_kde_and_temporal_graphs():

    # sample video
    vid = 'Stan_vid' + str(1) + '_1.csv'

    df_actual_arr = read_actual(vid)[:, 0] / 1000000
    df_synth_ori = read_synth(vid)[:, 0] / 1000000

    plot_kde_and_time(data_1=df_actual_arr,
                      data_2=df_synth_ori,
                      label_1='Actual',
                      label_2='Generated')

    return

draw_kde_and_temporal_graphs()

