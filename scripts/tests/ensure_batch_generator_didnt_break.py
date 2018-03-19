#! /usr/bin/env python3
"""Script to ensure batch_generator.py wasn't broken in the prettify process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
from itertools import product
import random

import tensorflow as tf
import regex as re

sys.path.append('../')
from batch_generator import BatchGenerator
from utils import data_utils, model_utils
from deep_quant import get_configs
from train import run_epoch


def set_configs(config, params):
    stride, forecast_n, min_unrollings, \
        max_unrollings, start_date, end_date = params

    config.key_field = 'gvkey'
    config.active_field = 'active'
    config.categorical_fields = 'sector'
    config.target_field = 'oiadpq_ttm'
    config.financial_fields = 'saleq_ttm-ltq_mrq'
    config.aux_fields = 'mom1m-mom9m'
    config.scale_field = 'mrkcap'
    config.datafile = 'open-dataset.dat'
    config.data_dir = 'datasets'
    config.nn_type = 'DeepMlpModel'
    config.optimizer = 'AdadeltaOptimizer'
    config.data_scaler = 'RobustScaler'
    config.passes = 1.0
    config.stride = stride
    config.forecast_n = forecast_n
    config.max_unrollings = max_unrollings
    config.min_unrollings = min_unrollings
    config.batch_size = 64
    config.validation_size = 0.30
    config.seed = 521
    config.max_epoch = 5
    config.early_stop = 5
    config.keep_prob = 1.0
    config.learning_rate = 0.6
    config.lr_decay = 0.95
    config.init_scale = 0.1
    config.max_grad_norm = 10.0
    config.num_layers = 1
    config.num_hidden = 128
    config.skip_connections = False
    config.input_dropout = False
    config.hidden_dropout = False
    config.rnn_lambda = 1.0
    config.target_lambda = 0.5
    config.cache_id = None


def load_train_valid_data(config, verbose):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    data_path = data_utils.get_data_path(config.data_dir, config.datafile)
    batches = BatchGenerator(data_path, config, verbose=verbose)
    train_data = batches.train_batches()
    valid_data = batches.valid_batches()
    return train_data, valid_data


def train_model(config, verbose=True):
    print("\nLoading training data ...")
    train_data, valid_data = load_train_valid_data(config, verbose)

    if config.start_date is not None:
        print("Training start date: ", config.start_date)
    if config.start_date is not None:
        print("Training end date: ", config.end_date)

    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False,
                               device_count={'GPU': 0})

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        if config.seed is not None:
            tf.set_random_seed(config.seed)

        print("\nConstructing model ...")
        model = model_utils.get_model(session, config, verbose=verbose)

        if config.data_scaler is not None:
            start_time = time.time()
            print("Calculating scaling parameters ...", end=' ')
            sys.stdout.flush()
            scaling_params = train_data.get_scaling_params(config.data_scaler)
            model.set_scaling_params(session, **scaling_params)
            print("done in %.2f seconds." % (time.time() - start_time))

        if config.early_stop is not None:
            print("Training will early stop without "
                  "improvement after %d epochs." % config.early_stop)

        train_history = list()
        valid_history = list()

        lr = model.set_learning_rate(session, config.learning_rate)

        train_data.cache(verbose=verbose)
        valid_data.cache(verbose=verbose)

        for i in range(config.max_epoch):
            (train_mse, valid_mse) = run_epoch(session, model, train_data,
                                               valid_data,
                                               keep_prob=config.keep_prob,
                                               passes=config.passes,
                                               verbose=verbose)
            if verbose:
                print(('Epoch: %d Train MSE: %.6f Valid MSE: %.6f Learning'
                       'rate: %.4f') % (i + 1, train_mse, valid_mse, lr))
                sys.stdout.flush()

            train_history.append(train_mse)
            valid_history.append(valid_mse)

            if re.match("Gradient|Momentum", config.optimizer):
                lr = model_utils.adjust_learning_rate(
                        session, model, lr, config.lr_decay, train_history)

    return train_history, valid_history


def get_results(config, params):
    set_configs(config, params)
    train_history, valid_history = train_model(config)
    return train_history, valid_history


# SETUP POSSIBLE PARAMETER SETTINGS
# stride: the space (in number of timesteps) between timesteps used as input
stride_possible = [3, 6, 12]

# forecast_n: the number of timesteps in the future that we want to forecast
forecast_n_possible = [1, 6, 12, 18, 24]

min_unrollings_possible = [4, 5]
max_unrollings_possible = [4, 5]

start_date_possible = [201003, None]
end_date_possible = [201603, None]

all_permutations = list(product(
        stride_possible,
        forecast_n_possible,
        min_unrollings_possible,
        max_unrollings_possible,
        start_date_possible,
        end_date_possible))

random.seed(521)
num_to_sample = round(len(all_permutations)*0.05)
permutations_to_test = random.sample(all_permutations, num_to_sample)


# GET OLD RESULTS
# import pickle
# config = get_configs()
# old_results = dict()
# for params in permutations_to_test:
#     train_history_old, valid_history_old = get_results(config, params)
#     old_results[params] = (train_history_old, valid_history_old)
#
# pickle.dump(old_results, open("old_results.p", "wb"))
# old_results = pickle.load(open("old_results.p", "rb"))

old_results = {
        (3,
         1,
         4,
         4,
         201003,
         201603):
        ([3.1978184853956897,
          1.322326654000093,
          0.9035426562476174,
          0.6996359891537195,
          0.5936778953143493],
         [4.134138479019506,
          2.319443346974187,
          1.4508533156194245,
          1.019519846922102,
          0.7487293909283016]),
        (3,
         6,
         5,
         4,
         201003,
         None):
        ([3.34677825601641,
          2.123125900499156,
          1.7224957678583623,
          1.5206467362478842,
          1.3730319909940178],
         [4.760814002582005,
          3.5863532282230333,
          2.8941688899483,
          2.6511312271452114,
          2.3828527642874766]),
        (3,
         12,
         5,
         5,
         201003,
         None):
        ([2.697466883296593,
          2.043354461320673,
          1.8409628040207842,
          1.743128954864348,
          1.6632784489339913],
         [4.396587286311299,
          3.5814356879511875,
          3.302431684154183,
          3.127725523910416,
          3.006866025168504]),
        (3,
         24,
         4,
         4,
         201003,
         201603):
        ([2.622411010303736,
          2.2869450992409788,
          2.128118497634403,
          2.031226139828856,
          1.9670384902113878],
         [4.654907550647169,
          3.94427818434009,
          3.775870106147922,
          3.715828199765672,
          3.6428392397933425]),
        (3,
         24,
         5,
         5,
         None,
         None):
        ([2.653948206798228,
          2.2739947743206965,
          2.108030183200845,
          2.017935846469609,
          1.9596255077786904],
         [4.818226793492801,
          4.1584382366251065,
          3.981553172493812,
          3.932283063080316,
          3.8310550770728415]),
        (6,
         6,
         5,
         4,
         201003,
         201603):
        ([4.227787200021662,
          2.865539264812322,
          2.2515673584556417,
          1.9565791840089577,
          1.765623027596642],
         [6.347486376523014,
          5.01349937574691,
          3.966867125417334,
          3.5791159378117827,
          3.2087683640570046]),
        (12,
         6,
         4,
         5,
         None,
         201603):
        ([3.9974851053158846,
          3.0197959484064447,
          2.38363022749082,
          2.00169857772539,
          1.7590336162039852],
         [6.031561853766725,
          5.2190383381062055,
          4.646765066863805,
          4.260529323173532,
          3.973861419177678]),
        (12,
         12,
         4,
         5,
         None,
         201603):
        ([3.2056531010389593,
          2.575051057178523,
          2.2323192400734744,
          2.00281726553085,
          1.8622731572436326],
         [5.226769493668491,
          4.808011446011628,
          4.55696475864705,
          4.422923823769804,
          4.421655669106239]),
        (12,
         12,
         5,
         4,
         None,
         None):
        ([3.96150355888348,
          3.2699184460299358,
          2.856294188011928,
          2.547771086253706,
          2.3329717899237026],
         [6.430221324475085,
          5.881209423893788,
          5.554736330860951,
          5.352340211047501,
          5.093019836163911]),
        (12,
         12,
         5,
         5,
         None,
         None):
        ([3.8440504205160115,
          3.109940127706995,
          2.6781926588768385,
          2.3988582167555306,
          2.21337173570271],
         [6.267808333381278,
          5.719518181144214,
          5.446060808955646,
          5.2554416222650495,
          4.975786738024383]),
        (12,
         18,
         5,
         4,
         None,
         None):
        ([3.5532536268807373,
          3.0859957949186745,
          2.8015768794963756,
          2.5648165680945683,
          2.3861978189685407],
         [6.995428636725922,
          6.381005937240544,
          6.1306613138998935,
          5.938919483275896,
          5.760625588536709]),
        (12,
         18,
         5,
         5,
         201003,
         None):
        ([3.2134198933266678,
          2.746327087856256,
          2.470416814279862,
          2.2641347865932264,
          2.1051878618219724],
         [6.205265772923101,
          5.6801021617003595,
          5.4819102492671785,
          5.339269188682684,
          5.207024006807849])}


# CHECK THAT EACH REMAINS THE SAME
config = get_configs()
new_results = dict()
for params in permutations_to_test:
    train_history_new, valid_history_new = get_results(config, params)
    new_results[params] = (train_history_new, valid_history_new)

assert(old_results == new_results)
