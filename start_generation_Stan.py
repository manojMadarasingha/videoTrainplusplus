#  Adversarial data generation via percentile distribution mapping
#  Copyright (c) 2021.
#  Authors : Shashika R. Muramudalige (shashika@colostate.edu), Anura P. Jayasumana (anura@colostate.edu), and
#  Haonan Wang (wanghn@stat.colostate.edu)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.

import wgan_generation_Stan as wgan_new
import pandas as pd
import numpy as np
import os
import time

path_out_name = 'synth'


def write_single_data_file(gen_data, path_out, category, file_name):
    write_data = []  # assume both parts have same batch size
    for d in range(len(gen_data)):
        final_arr = np.asarray(gen_data[d])
        write_data.append(final_arr)

    write_data = np.concatenate(write_data, axis=0)

    df = pd.DataFrame(data=np.array(write_data),
                      columns=[' addr1_packet', ' addr1_bytes', ' addr2_packet', ' addr2_bytes'])

    output_path = path_out + '/' + category + '/' + path_out_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df.to_csv(output_path + '/' + file_name,
              index=False)


# increase the bin size from 0.36s to the 0.36s*multiplier
def increase_bin_size(bin_divide, trace):
    new_trace = trace.reshape([-1, bin_divide, trace.shape[1]]).sum(1)

    return new_trace


if __name__ == '__main__':

    # factor used to split the 500 bins
    bin_divide = 4
    # location to the data inputs and outputs
    curr_work_dir = os.getcwd()
    path_in = curr_work_dir + '/data'
    path_out = curr_work_dir + '/data'

    category = 'Stan'

    file_names = [category + '_vid' + str(1) + '_' + str(1) + '.csv',
                  category + '_vid' + str(2) + '_' + str(1) + '.csv', ]

    for name in file_names:

        print(name)

        columns = [' addr1_packet', ' addr1_bytes', ' addr2_packet', ' addr2_bytes']
        trace_ori = pd.read_csv(path_in + '/' + category + '/actual/' + name)[columns].values

        if name.startswith(category):
            margin = int(500 // bin_divide)

            # increased_bin_size
            trace_ori = increase_bin_size(bin_divide=bin_divide, trace=trace_ori)

            gen_data, cos_sim_ori, num_of_epochs_ori, is_negtive_cosine_ori, cosine_list_ori, epoch_list_ori = wgan_new.main(
                name, category,
                margin, 0, 131,
                trace_ori)

            split_ori_str = ''
            for s in range(len(cosine_list_ori)):
                split_ori_str += str(cosine_list_ori[s]) + ','
                split_ori_str += str(epoch_list_ori[s]) + ','

            write_single_data_file(gen_data, path_out, category, name)
