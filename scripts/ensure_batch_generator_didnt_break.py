#! /usr/bin/env python3
"""Script to ensure batch_generator.py wasn't broken in the prettify process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

from itertools import product

import tensorflow as tf
import numpy as np
import regex as re
import pandas as pd

from prettified_batch_generator import BatchGenerator as PrettyBatchGenerator
from batch_generator import BatchGenerator  # this is the old one
import configs as configs
from utils import data_utils, model_utils
from deep_quant import get_configs
from train import pretty_progress, run_epoch


def set_configs(config, params):
    stride, forecast_n, min_unrollings, max_unrollings = params

    config.key_field='gvkey'
    config.active_field='active'
    config.categorical_fields='sector'
    config.target_field='oiadpq_ttm'
    config.financial_fields='saleq_ttm-ltq_mrq'
    config.aux_fields='mom1m-mom9m'
    config.scale_field='mrkcap'
    config.datafile='open-dataset.dat'
    config.data_dir='datasets'
    config.nn_type='DeepMlpModel'
    config.optimizer='AdadeltaOptimizer'
    config.data_scaler='RobustScaler'
    config.passes=1.0
    config.stride=stride
    config.forecast_n=forecast_n
    config.max_unrollings=max_unrollings
    config.min_unrollings=min_unrollings
    config.batch_size=64
    config.validation_size=0.30
    config.seed=521
    config.max_epoch=10
    config.early_stop=10
    config.keep_prob=1.0
    config.learning_rate=0.6
    config.lr_decay=0.95
    config.init_scale=0.1
    config.max_grad_norm=10.0
    config.num_layers=1
    config.num_hidden=128
    config.skip_connections=False
    config.input_dropout=False
    config.hidden_dropout=False
    config.rnn_lambda=1.0
    config.target_lambda=0.5
    config.cache_id=None


def load_all_data(config, pretty, verbose):
    """
    Returns all data as a BatchGenerator object.
    """
    data_path = data_utils.get_data_path(config.data_dir, config.datafile)
    if pretty:
        batches = PrettyBatchGenerator(data_path, config, verbose=verbose)
    else:
        batches = BatchGenerator(data_path, config, verbose=verbose)

    return batches


def load_train_valid_data(config, pretty, verbose):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    batches = load_all_data(config, pretty, verbose)
    
    train_data = batches.train_batches()
    valid_data = batches.valid_batches()
    
    return train_data, valid_data


def train_model(config, pretty=False, verbose=True):
    print("\nLoading training data ...")
    train_data, valid_data = load_train_valid_data(config, pretty, verbose)
    
    if config.start_date is not None:
        print("Training start date: ", config.start_date)
    if config.start_date is not None:
        print("Training end date: ", config.end_date)

    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False,
                               device_count = {'GPU': 0})
    
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        if config.seed is not None:
            tf.set_random_seed(config.seed)
            
        print("\nConstructing model ...")
        model = model_utils.get_model(session, config, verbose=verbose)
        
        if config.data_scaler is not None:
            start_time = time.time()
            print("Calculating scaling parameters ...", end=' '); sys.stdout.flush()
            scaling_params = train_data.get_scaling_params(config.data_scaler)
            model.set_scaling_params(session,**scaling_params)
            print("done in %.2f seconds."%(time.time() - start_time))
            
        if config.early_stop is not None:
            print("Training will early stop without "
                  "improvement after %d epochs." % config.early_stop)
            
        train_history = list()
        valid_history = list()
        
        lr = model.set_learning_rate(session, config.learning_rate)
        
        train_data.cache(verbose=verbose)
        valid_data.cache(verbose=verbose)
        
        for i in range(config.max_epoch):
            (train_mse, valid_mse) = run_epoch(session, model, train_data, valid_data,
                                               keep_prob=config.keep_prob, passes=config.passes,
                                               verbose=verbose)
            if verbose:
                print( ('Epoch: %d Train MSE: %.6f Valid MSE: %.6f Learning rate: %.4f') %
                      (i + 1, train_mse, valid_mse, lr) )
                sys.stdout.flush()
            
            train_history.append( train_mse )
            valid_history.append( valid_mse )
            
            if re.match("Gradient|Momentum",config.optimizer):
                lr = model_utils.adjust_learning_rate(session, model, lr, config.lr_decay, train_history )
                
    return train_history, valid_history


def is_same_result(config, params):
    set_configs(config, params)
    train_history_old, valid_history_old = train_model(config)
    train_history_pretty, valid_history_pretty = train_model(config, 
        pretty=True)
    is_same_result = (train_history_old == train_history_pretty) \
                     and (valid_history_old == valid_history_pretty)
    return is_same_result

## SETUP POSSIBLE PARAMETER SETTINGS
# stride: the space (in number of timesteps) between timesteps used as input
stride_possible = [3, 6, 12]

# forecast_n: the number of timesteps in the future that we want to forecast
forecast_n_possible = [1, 6, 12, 18, 24]

min_unrollings_possible = [4, 5]
max_unrollings_possible = [4, 5]

permutations = product(
        stride_possible, 
        forecast_n_possible,
        min_unrollings_possible,
        max_unrollings_possible)

## CHECK THAT EACH REMAINS THE SAME
config = get_configs()
test_results = [is_same_result(config, params) for params in permutations]
assert all(test_results)
