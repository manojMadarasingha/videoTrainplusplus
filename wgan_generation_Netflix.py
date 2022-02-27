#  Adversarial data generation via percentile distribution mapping
#  Copyright (c) 2021.
#  Authors : Shashika R. Muramudalige (shashika@colostate.edu), Anura P. Jayasumana (anura@colostate.edu), and
#  Haonan Wang (wanghn@stat.colostate.edu)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.

from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
from scipy import spatial
import logging

import os
import math
import logging
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)
# tf.logging.set_verbosity(tf.logging.ERROR)
CUDA_VISIBLE_DEVICES = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# For the wgan based data generation we use this script automatically switching between the similarity fucntions.
# for Netflix only

class WGAN:

    def __init__(self):
        self.matrix_rows = None
        self.matrix_cols = 4
        self.channels = 1
        self.matrix_shape = None
        self.latent_dim = 100
        self.input_data = []
        self.perc = None
        self.data_wgan = None
        self.gen_arr = []
        self.attr = []
        self.zero_percentiles = []

        self.cosine_similarity_list = []
        self.euc_distance_list = []
        self.epoch_list = []
        self.trace_list = []

        self.best_set_found = False
        self.is_negtive_cosine = False
        self.is_positive_euclead = False

        self.temp_epoch = 0
        self.best_cos_sim = 0.95
        self.best_euclead = 3.8

        # Dataset specific
        self.data_columns = 4

    def define_wgan_model(self):
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(np.prod(self.matrix_shape), activation='tanh'))
        model.add(Reshape(self.matrix_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()
        model.add(Input(shape=self.matrix_shape))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)

        model.summary()

        img = Input(shape=self.matrix_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128):

        graphs_train = self.data_wgan

        # Rescale -1 to 1
        graphs_train = 2 * graphs_train - 1.
        graphs_train = np.expand_dims(graphs_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            self.temp_epoch = epoch

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, graphs_train.shape[0], batch_size)
                imgs = graphs_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(size=(batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            if epoch >= self.epoch_at_sarting_generate_data and epoch % self.skip_val_data_generate == 0:
                self.get_cosin_sim_and_euc_dist(batch_size=1, no_of_images=200)
                self.epoch_list.append(epoch)

            # if the current cosine similairity of the traces synthesized using the
            # currently trained generator exceeds the 0.8 cosine similarity value, then take
            # that dataset as the final synthesized traces.
            if self.option == 0:
                if self.best_set_found or self.is_negtive_cosine:
                    break
            else:
                if self.best_set_found or self.is_positive_euclead:
                    break

        return

    def get_cosin_sim_and_euc_dist(self, batch_size, no_of_images):

        temp_gen_arr = []
        for g in range(no_of_images):
            batch_arr = []
            noise = np.random.normal(size=(batch_size, self.latent_dim))
            gen_graphs = self.generator.predict(noise)
            gen_graphs = 0.5 * gen_graphs + 0.5

            for i in range(self.matrix_rows):
                gen = []
                for j in range(self.matrix_cols):
                    gen.append(gen_graphs[0][i][j][0])
                batch_arr.append(gen)
            temp_gen_arr.append(batch_arr)

        write_arr = []

        for i in range(len(temp_gen_arr)):
            batch_arr = []
            for j in range(len(temp_gen_arr[i])):
                values = []

                for a in range(len(temp_gen_arr[i][j])):

                    arr = self.attr[a]
                    val = abs(temp_gen_arr[i][j][a]) * 100
                    v = np.percentile(arr, val)

                    if val < self.zero_percentiles[a]:
                        values.append(0)
                    else:
                        values.append(v)

                batch_arr.append(values)

            write_arr.append(batch_arr)

        # calculate the cosine similarity between the created traces and the actual traces.
        vec1 = self.input_data[:, -1]
        all_cos_sims = []
        all_ecu_dist = []
        # only consider the first 40 traces assuming that remaining can replicate
        for d in range(0, 40):
            data = write_arr[d]
            vec2 = np.asarray(data)[:, -1]
            if self.option == 0:
                cosine_similarity = 1 - spatial.distance.cosine(vec1, vec2)
                all_cos_sims.append(cosine_similarity)
            else:
                # n, bins, patches = plt.hist(vec1, 50, density=True)
                max_val = np.maximum(np.max(vec1), (np.max(vec2)))
                euc_dist = np.linalg.norm((vec1 / max_val) - (vec2 / max_val))
                all_ecu_dist.append(euc_dist)
        if self.option == 0:
            mean_cosine_val = np.mean(np.asarray(all_cos_sims))
        else:
            mean_euc_dist = np.mean(np.asarray(all_ecu_dist))

        self.trace_list.append(write_arr)

        # replaced the cosine value with the euclead value
        if self.option == 0:
            print('+++++++ ' + str(mean_cosine_val) + ' ++++++++')
            self.cosine_similarity_list.append(mean_cosine_val)
            if mean_cosine_val > self.best_cos_sim:
                self.best_set_found = True
                return
            if len(self.cosine_similarity_list) >= 6:
                self.is_negtive_cosine = self.check_gradient_of_cosin_val()
        else:
            print('+++++++ ' + str(mean_euc_dist) + ' ++++++++')
            self.euc_distance_list.append(mean_euc_dist)
            if mean_euc_dist < self.best_euclead:
                self.best_set_found = True
                return

            if len(self.euc_distance_list) >= 6:
                self.is_positive_euclead = self.check_gradient_of_euclead_val()

        return

    # check whether monotonically decreasing
    def check_gradient_of_cosin_val(self):

        last_5_cosine_values = np.asarray(self.cosine_similarity_list[-5:])
        diff_arr = np.diff(last_5_cosine_values)
        if np.all(diff_arr <= 0):
            return True
        else:
            return False

    # check whether monotonically increasing
    def check_gradient_of_euclead_val(self):

        last_5_euclead_values = np.asarray(self.euc_distance_list[-5:])
        diff_arr = np.diff(last_5_euclead_values)
        if np.all(diff_arr >= 0):
            return True
        else:
            return False

    def assign_to_data_shape(self):

        max_no_of_batches = 500
        self.matrix_rows = len(self.input_data)

        self.matrix_shape = (self.matrix_rows, self.matrix_cols, self.channels)
        self.data_wgan = np.zeros(shape=(max_no_of_batches, self.matrix_rows, self.matrix_cols))

        attr = []
        for i in range(len(self.input_data[0])):
            attr.append(np.sort(self.input_data[:, i]))

        for b in range(max_no_of_batches):

            mat = np.zeros(shape=(self.matrix_rows, self.matrix_cols))

            for i in range(len(self.input_data)):

                for j in range(0, len(self.input_data[i])):
                    arr = attr[j]
                    val = self.input_data[i][j]

                    percentile_no = stats.percentileofscore(arr, val, kind='weak')
                    if val == 0:
                        percentile_no = random.uniform(0, percentile_no)

                    mat[i][j] = percentile_no / 100

            self.data_wgan[b] = mat

    def read_file(self, path_in, file_name, category, margin, option):
        df = pd.read_csv(path_in + '/' + category + '/actual/' + file_name,
                         usecols=[' addr1_packet', ' addr1_bytes', ' addr2_packet', ' addr2_bytes'])
        df = df.replace(np.nan, 0)
        data = df.to_numpy(dtype='float')
        if option == 0:
            self.input_data = data[:margin]
        else:
            self.input_data = data[margin:]

        for i in range(self.data_columns):
            self.attr.append(np.sort(self.input_data[:, i]))

        for i in range(len(self.attr)):
            arr = self.attr[i]
            percentile_no = stats.percentileofscore(arr, 0, kind='weak')
            self.zero_percentiles.append(percentile_no)

    def draw_traces(self):
        plt.plot(self.input_data[:, 1], color='b', linewidth=1, label='Actual')
        plt.legend()
        plt.show()

    def generate_new_data(self, batch_size, no_of_images):
        for g in range(no_of_images):
            batch_arr = []
            noise = np.random.normal(size=(batch_size, self.latent_dim))
            gen_graphs = self.generator.predict(noise)
            gen_graphs = 0.5 * gen_graphs + 0.5

            for i in range(self.matrix_rows):
                gen = []
                for j in range(self.matrix_cols):
                    gen.append(gen_graphs[0][i][j][0])
                batch_arr.append(gen)
            self.gen_arr.append(batch_arr)

    def postprocess_data(self):
        write_arr = []

        for i in range(len(self.gen_arr)):
            batch_arr = []
            for j in range(len(self.gen_arr[i])):
                values = []

                for a in range(len(self.gen_arr[i][j])):

                    arr = self.attr[a]
                    val = abs(self.gen_arr[i][j][a]) * 100
                    v = np.percentile(arr, val)

                    if val < self.zero_percentiles[a]:
                        values.append(0)
                    else:
                        values.append(v)

                batch_arr.append(values)

            write_arr.append(batch_arr)

        return write_arr

    def assign_parameters(self, option):
        self.option = option
        if option == 0:
            self.epoch_at_sarting_generate_data = 70
            self.skip_val_data_generate = 5
        else:
            self.epoch_at_sarting_generate_data = 300
            self.skip_val_data_generate = 10


def main(path_in, file_name, category, margin, option, epochs):
    # strategy = tf.distribute.MirroredStrategy()
    try:
        aae = WGAN()
        aae.read_file(path_in, file_name, category, margin, option)
        aae.assign_parameters(option)
        aae.assign_to_data_shape()
        aae.define_wgan_model()
        #     Specify an valid GPU device
        with tf.device('/device:GPU:0'):
            aae.train(epochs=epochs, batch_size=32)
    except RuntimeError as e:
        print(e)

    # aae.generate_new_data(batch_size=1, no_of_images=20)
    # return aae.postprocess_data()
    # if the best set is already found return the latest created traces list
    if option == 0:
        if aae.best_set_found:
            print('=========== ' + str(aae.cosine_similarity_list[-1]) + ' ================')
            return aae.trace_list[-1], aae.cosine_similarity_list[-1], aae.epoch_list[-1], aae.is_negtive_cosine, \
                   aae.cosine_similarity_list, aae.epoch_list
        else:
            # find the best cosine similarity value and get the
            cos_sim = np.asarray(aae.cosine_similarity_list)
            max_cos_sim = np.nanmax(cos_sim)
            if not math.isnan(max_cos_sim):
                best_trace_set_ind = aae.cosine_similarity_list.index(max_cos_sim)
                print('=========== ' + str(max_cos_sim) + ' ================')
            else:
                best_trace_set_ind = -1
            return aae.trace_list[best_trace_set_ind], aae.cosine_similarity_list[
                best_trace_set_ind], aae.epoch_list[best_trace_set_ind], aae.is_negtive_cosine, \
                   aae.cosine_similarity_list, aae.epoch_list

    else:
        if aae.best_set_found:
            print('=========== ' + str(aae.euc_distance_list[-1]) + ' ================')
            return aae.trace_list[-1], aae.euc_distance_list[-1], aae.epoch_list[-1], aae.is_negtive_cosine, \
                   aae.euc_distance_list, aae.epoch_list
        else:
            # find the best euclidean distance value and get the
            euc_dist = np.asarray(aae.euc_distance_list)
            min_euc_dist = np.nanmin(euc_dist)
            if not math.isnan(min_euc_dist):
                best_trace_set_ind = aae.euc_distance_list.index(min_euc_dist)
                print('=========== ' + str(min_euc_dist) + ' ================')
            else:
                best_trace_set_ind = -1
            return aae.trace_list[best_trace_set_ind], aae.euc_distance_list[
                best_trace_set_ind], aae.epoch_list[best_trace_set_ind], aae.is_negtive_cosine, \
                   aae.euc_distance_list, aae.epoch_list
