#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

# #! /usr/bin/env python3
# Copyright 2016 Euclidean Technologies Management LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configs as configs
from utils.data_utils import load_train_valid_data


def get_configs():
    """
    Defines all configuration params passable to command line.
    """
    configs.DEFINE_string("datasource", 'big_datafile', 
                          "The source of the data.")
    configs.DEFINE_string("tkrlist", "big_tkrlist.csv", 
                          "The list of filters to use.")
    configs.DEFINE_string("datafile", 'big_datafile.dat', "a datafile name.")
    configs.DEFINE_string("mse_outfile", None, "A file to write mse values during predict phase.")
    configs.DEFINE_string("default_gpu", '', "The default GPU to use e.g., /gpu:0")
    configs.DEFINE_string("nn_type",'DeepRnnModel',"Model type")
    configs.DEFINE_string("active_field", 'active',"Key column name header for active indicator")
    configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
    configs.DEFINE_string("target_field", 'oiadpq_ttm',"Target column name header in datafile")
    configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
    configs.DEFINE_string("feature_fields", '',"shared input and target field names")
    configs.DEFINE_string("aux_input_fields", None,"non-target, input only fields")
    configs.DEFINE_string("data_dir",'',"The data directory")
    configs.DEFINE_string("model_dir",'',"Model directory")
    configs.DEFINE_string("rnn_cell",'gru',"lstm or gru")
    configs.DEFINE_integer("num_inputs", -1,"")
    configs.DEFINE_integer("num_outputs", -1,"")
    configs.DEFINE_integer("target_idx",None,"")
    configs.DEFINE_integer("min_unrollings",None,"Min number of unrolling steps")
    configs.DEFINE_integer("max_unrollings",None,"Max number of unrolling steps")
    # num_unrollings is being depricated by max_unrollings
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


def main(_):
    config = get_configs()
    
    train_data, valid_data = load_train_valid_data(config)

    train_data.cache(verbose=True)
    valid_data.cache(verbose=True)

if __name__ == "__main__":
    tf.app.run()
