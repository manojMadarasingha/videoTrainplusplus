#  Adversarial data generation via percentile distribution mapping
#  Copyright (c) 2021.
#  Authors : Shashika R. Muramudalige (shashika@colostate.edu), Anura P. Jayasumana (anura@colostate.edu), and
#  Haonan Wang (wanghn@stat.colostate.edu)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.

import wgan_generation_Netflix as wgan_new
import pandas as pd
import numpy as np
import os
import logging
import time
import argparse

path_out_name = 'synth'


def cal_margin(arr, consecutive_zeros):
    margin, count = 0, 0

    for i in range(1, len(arr)):
        now = 0 if arr[i] < 400 else arr[i]
        prev = 0 if arr[i] < 400 else arr[i]

        if now == prev == 0:
            count += 1
            if count == 1:
                margin = i + 1
            if count > consecutive_zeros:
                break
        else:
            count = 0
    return margin


def process_moving_window(byte_count):
    sum = 0
    cum = []
    x = []
    for i, v in enumerate(byte_count):
        sum += v
        x.append(i + 1)
        cum.append(sum)

    i = 0
    window_size = 20
    moving_averages = []
    while i < len(cum) - window_size + 1:
        this_window = cum[i: i + window_size]
        window_average = np.sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    g = np.gradient(moving_averages)
    consecutive_zeros = 5
    return cal_margin(g, consecutive_zeros)


# get the trace splitting point based on the margin value
def get_margin(path_in, category, file_name):
    df = pd.read_csv(path_in + '/' + category + '/actual/' + file_name,
                     usecols=[' addr1_packet', ' addr1_bytes', ' addr2_packet', ' addr2_bytes'])
    arr = df.to_numpy()
    byte_count_2 = arr[:, 3]  # addr2_bytes
    margin_2 = process_moving_window(byte_count_2)

    return margin_2


# commented this original version to exclude
def write_data_file(file_name, gen_data_1, gen_data_2, path_out):
    write_data = []  # assume both parts have same batch size
    for d in range(len(gen_data_1)):
        for i in range(len(gen_data_1[d])):
            write_data.append(gen_data_1[d][i])

        for i in range(len(gen_data_2[d])):
            write_data.append(gen_data_2[d][i])

    df = pd.DataFrame(data=np.array(write_data),
                      columns=[' addr1_packet', ' addr1_bytes', ' addr2_packet', ' addr2_bytes'])
    output_path = path_out + '/' + category + '/' + path_out_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df.to_csv(path_out + '/' + category + '/' + path_out_name + '/' + file_name,
              index=False)


if __name__ == '__main__':

    category = 'Netflix'

    # location to the data inputs and outputs
    curr_work_dir = os.getcwd()
    path_in = curr_work_dir + '/data'
    path_out = curr_work_dir + '/data'

    file_names = [category + '_vid' + str(1) + '_' + str(1) + '.csv',
                  category + '_vid' + str(2) + '_' + str(1) + '.csv', ]

    for name in file_names:
        print(name)

        if name.startswith(category):

            margin = get_margin(path_in, category, name)
            # commented to run only the second part of the trace.
            gen_data_1, cos_sim_first, num_of_epochs_first, is_negtive_cosine_first, cosine_list_1, epoch_list_1 = wgan_new.main(
                path_in,
                name, category,
                margin, 0, 301)  # first part
            gen_data_2, cos_sim_second, num_of_epochs_second, is_negtive_cosine_second, cosine_list_2, epoch_list_2 = wgan_new.main(
                path_in,
                name, category,
                margin, 1, 401)  # second part

            split_1_str = ''
            for s in range(len(cosine_list_1)):
                split_1_str += str(cosine_list_1[s]) + ','
                split_1_str += str(epoch_list_1[s]) + ','

            split_2_str = ''
            for s in range(len(cosine_list_2)):
                split_2_str += str(cosine_list_2[s]) + ','
                split_2_str += str(epoch_list_2[s]) + ','

            # commented to exclude the first part
            write_data_file(name, gen_data_1, gen_data_2, path_out)
