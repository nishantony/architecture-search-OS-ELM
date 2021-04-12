#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual
Author: Leonardo Lucio Custode
Date: 17/09/2020
"""
import pandas as pd
from abc import abstractmethod

from input_creator import input_gen
from network import network_fit

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    '''
    TODO: Consider hyperparameters of ELM instead of the number of neurons in hidden layers of MLPs. 
    Class for EA Task
    '''
    def __init__(self, dp, sequence_length, method, sensor_drop,
                 model_path, epochs, batch, visualize):
        self.dp = dp
        self.sequence_length = sequence_length
        self.method = method
        self.sensor_drop = sensor_drop
        self.model_path = model_path
        self.epochs = epochs
        self.batch = batch
        self.visualize = visualize


    def get_n_parameters(self):
        if self.method == 'non':
            return 2
        elif self.method == 'sfa':
            return 4
        elif self.method == 'pca':
            return 3

    def get_parameters_bounds(self):
        if self.method == 'non':
            bounds = [
                (10, 100),
                (10, 100),
            ]
        elif self.method == 'sfa':
            bounds = [
                (10, 100),
                (2, 26),
                (10, 100),
                (10, 100),
            ]
        elif self.method == 'pca':
            bounds = [
                (10, 100),
                (10, 100),
                (10, 100),
            ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        # print ("genotype", genotype)
        # print ("len(genotype)", len(genotype))

        """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
        data_class = input_gen(data_path_list=self.dp, sequence_length=self.sequence_length,
                               sensor_drop=self.sensor_drop, visualize=self.visualize)
        train_samples, label_array_train, test_samples, label_array_test = data_class.concat_vec()


        if self.method == 'non':
            mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                                   self.model_path,
                                   n_hidden1=genotype[0],
                                   n_hidden2=genotype[1] if genotype[1] < genotype[0] else genotype[0])

        elif self.method == 'sfa':
            train_samples,  test_samples = data_class.sfa(
                train_vec_samples=train_samples,
                test_vec_samples=test_samples,
                n_components=genotype[0],
                n_bins=genotype[1],
                alphabet='ordinal')
            mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                                   self.model_path,
                                   n_hidden1=genotype[2],
                                   n_hidden2=genotype[3] if genotype[3] < genotype[2] else genotype[2])



        elif self.method == 'pca':
            train_samples, test_samples = data_class.pca(
                train_vec_samples=train_samples,
                test_vec_samples=test_samples,
                n_components=genotype[0])
            mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                                   self.model_path,
                                   n_hidden1=genotype[1],
                                   n_hidden2=genotype[2] if genotype[2] < genotype[1] else genotype[1])



        fitness = mlps_net.train_net(epochs=self.epochs, batch_size=self.batch)


        return fitness

