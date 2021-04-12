'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from input_creator import input_gen
from network import network_fit

from task import SimpleNeuroEvolutionTask
from ea import GeneticAlgorithm

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# random seed predictable
jobs = 1
seed = 0
random.seed(seed)
np.random.seed(seed)

current_dir = os.path.dirname(os.path.abspath(__file__))

## Dataset path
train_FD001_path = current_dir +'/cmapss/train_FD001.csv'
test_FD001_path = current_dir +'/cmapss/test_FD001.csv'
RUL_FD001_path = current_dir+'/cmapss/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

train_FD002_path = current_dir +'/cmapss/train_FD002.csv'
test_FD002_path = current_dir +'/cmapss/test_FD002.csv'
RUL_FD002_path = current_dir +'/cmapss/RUL_FD002.txt'
FD002_path = [train_FD002_path, test_FD002_path, RUL_FD002_path]

train_FD003_path = current_dir +'/cmapss/train_FD003.csv'
test_FD003_path = current_dir +'/cmapss/test_FD003.csv'
RUL_FD003_path = current_dir +'/cmapss/RUL_FD003.txt'
FD003_path = [train_FD003_path, test_FD003_path, RUL_FD003_path]

train_FD004_path =current_dir +'/cmapss/train_FD004.csv'
test_FD004_path = current_dir +'/cmapss/test_FD004.csv'
RUL_FD004_path = current_dir +'/cmapss/RUL_FD004.txt'
FD004_path = [train_FD004_path, test_FD004_path, RUL_FD004_path]

## Assign columns name
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
col_rul = ['RUL_truth']

## Read csv file to pandas dataframe
FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
dp_str = ["none", "FD001", "FD002", "FD003", "FD004"]

## temporary model path for NN
model_path = current_dir +'/temp_net.h5'
# Log file path of EA in csv
directory_path = current_dir + '/EA_log'


def recursive_clean(directory_path):
    """clean the whole content of :directory_path:"""
    if os.path.isdir(directory_path) and os.path.exists(directory_path):
        files = glob.glob(directory_path + '*')
        for file_ in files:
            if os.path.isdir(file_):
                recursive_clean(file_ + '/')
            else:
                os.remove(file_)



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='RPs creator')
    parser.add_argument('-i', type=int, help='Input sources', required=True)
    parser.add_argument('-l', type=int, default=10, help='sequence length')
    parser.add_argument('--method', type=str, default='non', help='data representation:non, sfa or pca')
    parser.add_argument('--visualize', type=str, default='yes', help='visualize rps.')
    parser.add_argument('--epochs', type=int, default=1000, required=False, help='number epochs for network training')
    parser.add_argument('--batch', type=int, default=700, required=False, help='batch size of BPTT training')
    parser.add_argument('--verbose', type=int, default=0, required=False, help='Verbose TF training')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on cpu or cuda.')

    args = parser.parse_args()


    dp = FD_path[args.i]
    subdataset = dp_str[args.i]
    sequence_length = args.l
    device = args.device
    method = args.method
    epochs = args.epochs
    batch = args.batch
    verbose = args.verbose

    visualize = args.visualize
    if visualize == 'yes':
        visualize = True
    elif visualize == 'no':
        visualize = False

    ## Parameters for the GA
    pop_size = args.pop
    n_generations = args.gen
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"
    sel_op = "best"
    other_args = {
        'mut_gene_probability': 0.3  # 0.1
    }



    # Sensors not to be considered (those that do not disclose any pattern in their ts)
    sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

    start = time.time()

    print("Dataset: ", subdataset)
    print("Seq_len: ", sequence_length)
    print("Method: ", method)

    # Save log file of EA in csv
    recursive_clean(directory_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    mutate_log_path = 'EA_log/mute_log_%s_%s_%s_%s_%s_%s.csv' % (
        subdataset, sequence_length, method, epochs, pop_size, n_generations)
    if method == 'non':
        mutate_log_col = ['idx', 'params_1', 'params_2', 'fitness', 'gen']
    elif method == 'sfa':
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'fitness', 'gen']
    elif method == 'pca':
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'fitness', 'gen']
    mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
    mutate_log_df.to_csv(mutate_log_path, index=False)


    def log_function(population, gen, mutate_log_path = mutate_log_path):
        for i in range(len(population)):
            if population[i] == []:
                "non_mutated empty"
                pass
            else:
                # print ("i: ", i)
                population[i].append(population[i].fitness.values[0])
                population[i].append(gen)

        temp_df = pd.DataFrame(np.array(population), index=None)
        temp_df.to_csv(mutate_log_path, mode='a', header=None)
        print("population saved")
        return

    start = time.time()

    # Assign & run EA
    task = SimpleNeuroEvolutionTask(
        dp=dp,
        sequence_length=sequence_length,
        method=method,
        sensor_drop=sensor_drop,
        model_path=model_path,
        epochs=epochs,
        batch=batch,
        visualize=visualize
    )

    # aic = task.evaluate(individual_seed)

    ga = GeneticAlgorithm(
        task=task,
        population_size=pop_size,
        n_generations=n_generations,
        cx_probability=cx_prob,
        mut_probability=mut_prob,
        crossover_operator=cx_op,
        mutation_operator=mut_op,
        selection_operator=sel_op,
        jobs=jobs,
        log_function=log_function,
        **other_args
    )

    pop, log, hof = ga.run()

    print("Best individual:")
    print(hof[0])

    # Save to the txt file
    # hof_filepath = tmp_path + "hof/best_params_fn-%s_ps-%s_ng-%s.txt" % (csv_filename, pop_size, n_generations)
    # with open(hof_filepath, 'w') as f:
    #     f.write(json.dumps(hof[0]))

    print("Best individual is saved")
    end = time.time()
    print("EA time: ", end - start)



    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
    print ("Evaluate the best individual")
    data_class = input_gen(data_path_list=dp, sequence_length=sequence_length,
                           sensor_drop= sensor_drop, visualize=visualize, test=True)
    train_samples, label_array_train, test_samples, label_array_test = data_class.concat_vec()
    print ("train_samples.shape: ", train_samples.shape) # shape = (samples, sensors, concat_vec)
    print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
    print ("test_samples.shape: ", test_samples.shape) # shape = (samples, sensors, concat_vec)
    print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

    if method == 'non':
        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               model_path = model_path, n_hidden1=hof[0][0], n_hidden2=hof[0][1], verbose=verbose)

    elif method == 'sfa':
        train_samples, test_samples = data_class.sfa(train_samples, test_samples, n_components=hof[0][0], n_bins=hof[0][1])
        print ("train_samples.shape: ", train_samples.shape) # shape = (samples, sensors, height, width)
        print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
        print ("test_samples.shape: ", test_samples.shape) # shape = (samples, sensors, height, width)
        print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               model_path = model_path, n_hidden1=hof[0][2], n_hidden2=hof[0][3], verbose=verbose)

    elif method == 'pca':
        train_samples, test_samples = data_class.pca(train_samples, test_samples, n_components=hof[0][0])
        print ("train_samples.shape: ", train_samples.shape) # shape = (samples, sensors, height, width)
        print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
        print ("test_samples.shape: ", test_samples.shape) # shape = (samples, sensors, height, width)
        print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

        mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                               model_path = model_path, n_hidden1=hof[0][1], n_hidden2=hof[0][2], verbose=verbose)




    rms, score  = mlps_net.test_net(epochs=epochs, batch_size= batch, lr= 1e-05, plotting=True)


    print(subdataset + " test RMSE: ", rms)
    print(subdataset + " test Score: ", score)



if __name__ == '__main__':
    main()
