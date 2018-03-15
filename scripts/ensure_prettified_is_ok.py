#! /usr/bin/env python3
"""Script to ensure batch_generator.py wasn't broken in the prettify process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

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


def get_configs():
    """
    Defines all configuration params passable to command line.
    """
    configs.DEFINE_string("categorical_fields",None,"A comma-separated list of categorical fields.")
    configs.DEFINE_string("name",'none',"A name for the config.")
    configs.DEFINE_string("datafile", 'open-dataset.dat', "a datafile name.")
    configs.DEFINE_string("mse_outfile", None, "A file to write mse values during predict phase.")
    configs.DEFINE_string("default_gpu", '', "The default GPU to use e.g., /gpu:0")
    configs.DEFINE_string("nn_type",'DeepRnnModel',"Model type")
    configs.DEFINE_string("active_field", 'active',"Key column name header for active indicator")
    configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
    configs.DEFINE_string("target_field", 'oiadpq_ttm',"Target column name header in datafile")
    configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
    configs.DEFINE_string("financial_fields", '',"shared input and target field names")
    configs.DEFINE_string("aux_fields", None,"non-target, input only fields")
    configs.DEFINE_string("data_dir",'',"The data directory")
    configs.DEFINE_string("model_dir",'',"Model directory")
    configs.DEFINE_string("rnn_cell",'gru',"lstm or gru")
    configs.DEFINE_integer("num_inputs", -1,"")
    configs.DEFINE_integer("num_outputs", -1,"")
    configs.DEFINE_integer("target_idx",None,"")
    configs.DEFINE_integer("min_unrollings",None,"Min number of unrolling steps")
    configs.DEFINE_integer("max_unrollings",None,"Max number of unrolling steps")
    configs.DEFINE_integer("min_years",None,"Alt to min_unrollings")
    configs.DEFINE_integer("pls_years",None,"Alt min_unrollings and max_unrollings")
    configs.DEFINE_integer("num_unrollings",4,"Number of unrolling steps")
    configs.DEFINE_integer("stride",12,"How many steps to skip per unrolling")
    configs.DEFINE_integer("forecast_n",12,"How many steps to forecast into the future")
    configs.DEFINE_integer("batch_size",1,"Size of each batch")
    configs.DEFINE_integer("num_layers",1, "Numer of RNN layers")
    configs.DEFINE_integer("num_hidden",10,"Number of hidden layer units")
    configs.DEFINE_float("init_scale",0.1, "Initial scale for weights")
    configs.DEFINE_float("max_grad_norm",10.0,"Gradient clipping")
    configs.DEFINE_integer("start_date",None,"First date to train on as YYYYMM")
    configs.DEFINE_integer("end_date",None,"Last date to train on as YYYYMM")
    configs.DEFINE_float("keep_prob",1.0,"Keep probability for dropout")
    configs.DEFINE_boolean("train",True,"Train model otherwise inference only")
    configs.DEFINE_boolean("input_dropout",False,"Do dropout on input layer")
    configs.DEFINE_boolean("hidden_dropout",False,"Do dropout on hidden layers")
    configs.DEFINE_boolean("rnn_dropout",False,"Do dropout on recurrent connections")
    configs.DEFINE_boolean("skip_connections",False,"Have direct connections between input and output in MLP")
    configs.DEFINE_boolean("use_cache",True,"Load data for logreg from cache (vs processing from batch generator)")
    configs.DEFINE_boolean("pretty_print_preds",False,"Print predictions in tabular format with inputs, targets, and keys")
    configs.DEFINE_boolean("scale_targets",True,"")
    configs.DEFINE_string("data_scaler",None,'sklearn scaling algorithm or None if no scaling')
    configs.DEFINE_string("optimizer", 'GradientDescentOptimizer', 'Any tensorflow optimizer in tf.train')
    configs.DEFINE_string("optimizer_params",None, 'Additional optimizer params such as momentum')
    configs.DEFINE_float("learning_rate",0.6,"The initial starting learning rate")
    configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
    configs.DEFINE_float("validation_size",0.0,"Size of validation set as %, ie. .3 = 30% of data")
    configs.DEFINE_float("passes",1.0,"Passes through day per epoch")
    configs.DEFINE_float("target_lambda",0.5,"How much to weight last step vs. all steps in loss")
    configs.DEFINE_float("rnn_lambda",0.5,"How much to weight last step vs. all steps in loss")
    configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
    configs.DEFINE_integer("early_stop",None,"Early stop parameter")
    configs.DEFINE_integer("seed",None,"Seed for deterministic training")
    configs.DEFINE_integer("cache_id",None,"A unique experiment key for traking a cahce")
 
    c = configs.ConfigValues()
    
    if c.min_unrollings is None:
        c.min_unrollings = c.num_unrollings
    
    if c.max_unrollings is None:
        c.max_unrollings = c.num_unrollings

    if c.min_years is not None:
        c.min_unrollings = c.min_years * ( 12 // c.stride )
        if c.pls_years is None:
            c.max_unrollings = c.min_unrollings
        else:
            c.max_unrollings = (c.min_years+c.pls_years) * ( 12 // c.stride )

    # optimizer_params is a string of the form "param1=value1,param2=value2,..."
    # this maps it to dictionary { param1 : value1, param2 : value2, ...}
    if c.optimizer_params is None:
        c.optimizer_params = dict()
    else:
        args_list = [p.split('=') for p in c.optimizer_params.split(',')]
        params = dict()
        for p in args_list:
            params[p[0]] = float(p[1])
        c.optimizer_params = params
        assert('learning_rate' not in c.optimizer_params)

    return c


def set_configs(config, params):
    stride, forecast_n, min_unrollings, max_unrollings = params

    print("stride: {}".format(stride))
    print("forecast_n: {}".format(forecast_n))
    print("min_unrollings: {}".format(min_unrollings))
    print("max_unrollings: {}".format(max_unrollings))

    config.default_gpu='/gpu:0'
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
                               log_device_placement=False)
    
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


def is_same_result_for_params(params):
    set_configs(config, params)
    train_history_old, valid_history_old = train_model(config)
    train_history_pretty, valid_history_pretty = train_model(config, 
        pretty=True)
    is_same_result = (train_history_old == train_history_pretty) \
                     and (valid_history_old == valid_history_pretty)
    return is_same_result


config = get_configs()

# Default setting
stride = 12
forecast_n = 12
min_unrollings = 5
max_unrollings = 5
default_params = stride, forecast_n, min_unrollings, max_unrollings

test_results = list()
# Default test
test_results.append(is_same_result_for_params(default_params))

# Another test
test_results.append(is_same_result_for_params((12, 12, 4, 5)))
