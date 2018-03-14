# Copyright 2016 Euclidean Technologies Management LLC  All Rights Reserved.
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

import os
import time
import sys
import random
import pickle
import hashlib

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

_MIN_SEQ_NORM = 1.0
DEEP_QUANT_ROOT = os.environ['DEEP_QUANT_ROOT']
DATASETS_PATH = os.path.join(DEEP_QUANT_ROOT, 'datasets')

class BatchGenerator(object):
    """
    BatchGenerator takes a data file and returns an object with a next_batch()
    function. The next_batch() function yields a batch of data sequences from
    the datafile whose shape is specified by config.batch_size and
    config.max_unrollings.
    """
    def __init__(self, datpath, config, validation=True, require_targets=True,
                     data=None, verbose=True):
        """
        Init a BatchGenerator.
        Data is loaded as a Pandas DataFrame and stored in the `_data`
        attribute.
        Sequences of length config.max_unrollings are identified; the start and
        end indices of these sequences are stored in the _start_indices and
        _end_indices attributes of our BatchGenerator object, respectively. 
        (These are both lists of equal length that indicate the indices of the
        start and of the end of the sequence, within the _data DataFrame.)
        Indices are set up to access config.batch_size of these sequences as a
        batch; these are set up in such way that these sequences are "far apart"
        from each other in _data (they're num_batches rows away from each other,
        more specifically). This way it's unlikely that more than one sequence
        corresponds to the same company in a batch.
        """
        def init_data_attribute(self, datpath, config, data):
            """
            Reads .dat file at `datpath`, gets rid of excess timesteps if
            necessary.
            
            Also encodes each categorial covariate (as specified by
            `config.categorical_fields`) into its corresponding one-hot
            representation (as specified by its `field-encoding.dat` file), and
            appends this representation to the right of the `_data` attribute.
            """
            def encode_cat_attrib(data, cat_attrib):
                """
                Gets one-hot representation of the categorical attribute under
                `cat_attrib` column of `self._data`, appends that at
                right-end of `self._data` dataframe, populates
                `self._onehot_colnames` and `self._onehot_colixs`.
                """
                # Load encoding file as Pandas DataFrame
                encoding_file = "{}-encoding.dat".format(cat_attrib.lower())
                encoding_path = os.path.join(DATASETS_PATH, encoding_file)
                encoding_df = pd.read_csv(encoding_path, sep=' ')

                # Get categories under `cat_attrib` column as Numpy array of
                # codes (these codes are integers as specified in encoding file)
                cat_enc = LabelEncoder()
                cat_enc.fit(encoding_df[cat_attrib].values)
                categories = data[cat_attrib].values
                codes = cat_enc.transform(categories).reshape(-1, 1)

                # Get one-hot representation of each example's `cat_attrib`
                onehot_enc = OneHotEncoder(n_values=len(cat_enc.classes_))
                onehot_vecs = onehot_enc.fit_transform(codes).toarray()
                onehot_colnames = ['is_' + lev for lev in cat_enc.classes_]

                # Write all one-hot columns on the right side
                _, m = data.shape
                codes_df = pd.DataFrame(onehot_vecs,columns=onehot_colnames)
                data = pd.concat([data, codes_df], axis=1)

                # Store the column indices
                onehot_colixs = list(range(m, m + onehot_enc.n_values))
                
                return data, onehot_colixs

            self._aux_colixs = list()
            # Load data if necessary
            if data is None:
                data = pd.read_csv(datpath, sep=' ',
                                   dtype={config.key_field: str})

                # Encode categorical covariates
                cat_fields = config.categorical_fields
                cat_attribs = cat_fields.split(',') \
                              if cat_fields is not None else []

                for cat_attrib in cat_attribs:
                    data, onehot_colixs = encode_cat_attrib(data, cat_attrib)
                    self._aux_colixs += onehot_colixs

            # Get rid of excess dates
            data = data.drop(data[data['date'] < config.start_date].index)
            data = data.drop(data[data['date'] > config.end_date].index)

            # Store attributes
            self._data = data
            self._data_len = len(data)
            assert self._data_len

            return

        def init_column_indices(self, config):
            """
            # TODO: rewrite this docstring
            Sets up column-index-related attributes and adds a few items to the
            config.
            Column-index-related attributes:
              * _feature_indices: A list housing the column numbers of the features,
                                  where features are as specified by 
                                  config.financial_fields.
              * _aux_indices: A list housing the column numbers of the auxilary
                              covariates, where these auxilary covariates are
                              specified by config.aux_fields.
              * _input_names: A list housing the names of the columns of the 
                                features _and_ of the auxilary covariates.
              * _num_inputs: The total number of covariates used as input (so both
                             those that are specified in config.financial_fields and
                             those that are specified in config.aux_fields).
              * _key_idx: The column index of what should be used as a unique
                          identifier of each company (the index of gvkey, for 
                          example).
              * _active_idx: The column index of the field that lets us know whether
                             a company was actively trading during a specific point
                             in time or not.
              * _date_idx: The column index of the date field.
            Items added to the config:
              * num_inputs: same as the _num_inputs attribute
              * num_ouptus: num_inputs minus the number of aux covariates.
              * target_idx: index of target variable within the list of features, if
                            target is specified by config.
            """
            def get_colixs_from_colname_range(data, colname_range):
                """
                Returns indexes of columns of data that are in the range of
                `names`, inclusive. `names` should be a string with the
                following format: start_column_name-end_column_name
                (saleq_ttm-ltq_mrq, for example).
                """
                if colname_range is None:
                    colixs = []
                else:
                    assert 0 < colname_range.find('-') < len(colname_range)-1
                    first, last = colname_range.split('-')
                    start_ix = list(data.columns.values).index(first)
                    end_ix = list(data.columns.values).index(last)
                    assert start_ix >= 0
                    assert start_ix <= end_ix
                    colixs = list(range(start_ix, end_ix+1))
                return colixs
            
            def np_array_index(arr, value):
                """
                Replicates the Python list's `index` method (that is, it returns the
                first appearance of value in the array
                
                Raises `ValueError` if `value` is not present in `arr`.
                """
                index = None
                for i, element in enumerate(arr):
                    if element == value:
                        index = i
                        break

                if index is None:
                    raise ValueError("{} is not in arr.".format(value))

                return index

            assert config.financial_fields
            # Set up financials column indices and auxiliaries column indices
            self._fin_colixs = get_colixs_from_colname_range(
                    self._data, config.financial_fields)

            self._aux_colixs += get_colixs_from_colname_range(
                    self._data, config.aux_fields)

            # Set up other attributes
            colnames = self._data.columns.values
            self._key_idx = np_array_index(colnames, config.key_field)
            self._keys = self._data[config.key_field].tolist()
            self._date_idx = np_array_index(colnames, 'date')  # TODO: make a config
            self._dates = self._data['date'].tolist()
            self._active_idx = np_array_index(colnames, config.active_field)
            self._normalizer_idx = np_array_index(colnames, config.scale_field)

            # Set up input-related attributes
            self._input_names = list(colnames[self._fin_colixs\
                                              + self._aux_colixs])
            self._num_inputs = config.num_inputs = len(self._input_names)

            # Set up target index
            idx = np_array_index(colnames, config.target_field)
            if config.target_field == 'target':
                config.target_idx = 0
                self._num_outputs = config.num_outputs = 1
                self._price_target_idx = idx
            else:
                config.target_idx = idx - self._fin_colixs[0]
                self._num_outputs = config.num_outputs = self._num_inputs \
                                                         - len(self._aux_colixs)
                self._price_target_idx = -1

            assert(config.target_idx >= 0)

            # Set up fin_inputs attribute and aux_inputs attribute
            self._fin_inputs = self._data.iloc[:, self._fin_colixs].as_matrix()
            self._aux_inputs = self._data.iloc[:, self._aux_colixs].as_matrix()

        def init_validation_set(self, config, validation, verbose=True):
            """
            Sets up validation set, if necessary. 
            Creates the _validation_set attribute, which is a set housing the
            keys (unique identifier, such as gvkey) of the companies that should
            be used for validation.
            """
            # Setup the validation data
            self._validation_set = set()

            if validation is True:
                if config.seed is not None:
                    if verbose is True:
                        print("\nSetting random seed to " + str(config.seed))
                    random.seed(config.seed)
                    np.random.seed(config.seed)

                # get number of keys
                keys = sorted(set(self._data[config.key_field]))
                sample_size = int(config.validation_size * len(keys))
                sample = random.sample(keys, sample_size)
                self._validation_set = set(sample)

                if verbose is True:
                    print("Num training entities: %d"%(len(keys) - sample_size))
                    print("Num validation entities: %d"%sample_size)
            return

        def init_batch_cursor(self, config, require_targets=True, verbose=True):
            """
            Sets up indexes into the sequences.  First identifies start and end
            points of sequences (stored as _start_indices and _end_indices).
            Then sets up two cursors: 
              (1) _index_cursor, which is a cursor of 
                  equally-spaced indices into the dataset. Here, each index
                  points to a sequence (which can be determined fully using
                  _data, _start_indices, and _end_indices).  There will be
                  config.batch_size indices in this list.  
              (2) _batch_cursor, which keeps track of the batch that we're
                  working with. (This is just an int that changes as we go
                  through the dataset in batches.)
            Note that the number of batches is dictated by the number of
            sequences available and the user-defined size of each batch (as
            specified in config.batch_size). (The number of sequences available
            in turn depends on the length of those sequences,
            config.max_unrollings as well as the size of the dataset).
            Here, an attribute called _batch_cache is also created. This is a
            list of size num_batches that will house the contents of each batch
            once they're cached.
            Lastly, an attribute called _init_index_cursor is also created. This
            is simply a copy of _index_cursor in its original state, which will
            allow us to go back to the start if we need to once _index_cursor
            has changed.
            """
            def store_sequence_start_end_indices(self, data, stride, 
                    forecast_n, min_steps, max_steps, require_targets, config):
                """
                Populates the `_start_indices` and `_end_indices` attributes
                with the start and end index of each possible sequence in
                `data`, where elements of that sequence are `stride` timesteps
                apart, and each sequence has at least `min_steps` BLAH BLAH.
                """
                start_date = config.start_date if config.start_date is not None\
                                                                    else 100001
                last_key = ""
                cur_length = 1

                for i in range(self._data_len):
                    key = data.iat[i, self._key_idx]
                    if i+forecast_n < len(data):
                        pred_key = data.iat[i+forecast_n, self._key_idx]
                    else:
                        pred_key = ""

                    active = True if int(data.iat[i,self._active_idx]) else False
                    date = data.iat[i,self._date_idx]

                    if key != last_key:
                        cur_length = 1

                    if ((cur_length >= min_steps)
                         and (active is True)
                         and (date >= start_date)):
                        # If targets are not required, we don't need the future
                        # sequences to be there, otherwise we do
                        seq_len = min(cur_length-(cur_length-1) % stride, 
                                      max_steps)
                        if (not require_targets) or (key == pred_key):
                            self._start_indices.append(i-seq_len+1)
                            self._end_indices.append(i)
                    cur_length += 1
                    last_key = key

                return 

            self._start_indices = list()
            self._end_indices = list()
            
            min_steps = self._stride*(self._min_unrollings-1)+1
            max_steps = self._stride*(self._max_unrollings-1)+1
            store_sequence_start_end_indices(self, self._data, self._stride, 
                self._forecast_n, min_steps, max_steps, require_targets, config)

            if verbose is True:
                print("Num. of sequences: %d"%(len(self._start_indices)))

            # Create a cursor of equally spaced indices into the dataset. Each
            # index in the cursor points to one sequence in a batch and is used
            # to keep track of where we are in the dataset.
            self._num_batches = len(self._start_indices) // self._batch_size
            self._index_cursor = [offset*self._num_batches for offset in \
                                  range(self._batch_size)]
            self._init_index_cursor = self._index_cursor[:]
            self._batch_cache = [None]*self._num_batches
            self._batch_cursor = 0

        self._scaling_feature = config.scale_field
        self._scaling_params = None
        self._start_date = config.start_date
        self._end_date = config.end_date
        self._max_unrollings = config.max_unrollings
        self._min_unrollings = config.min_unrollings
        self._stride = config.stride
        self._forecast_n = config.forecast_n
        self._batch_size = config.batch_size

        assert self._stride >= 1
        
        ### INITIALIZE DATA ###
        init_data_attribute(self, datpath, config, data)
        init_column_indices(self, config)
        init_validation_set(self, config, validation, verbose)

        ### INITIALIZE CURSORS ###
        init_batch_cursor(self, config, require_targets, verbose)

        self._config = config # save this around for train_batches() method

    def _get_normalizer(self, end_idx):
        val = max(self._data.iat[end_idx, self._normalizer_idx], 
                  _MIN_SEQ_NORM)
        return val

    def _get_batch_normalizers(self):
        """
        Returns an np.array housing the normalizers (scalers) by which
        the inputs of the current sequence should be scaled (this is
        specified by config.scale_field).
        """
        v_get_normalizer = np.vectorize(self._get_normalizer)
        end_idxs = np.array(self._end_indices)[self._index_cursor]
        return v_get_normalizer(end_idxs)

    def _get_feature_vector(self, end_idx, cur_idx):
        if cur_idx < self._data_len:
            s = self._get_normalizer(end_idx)
            assert(s>0)
            x = self._fin_inputs[cur_idx]
            y = np.divide(x,s)
            y_abs = np.absolute(y).astype(float)
            return np.multiply(np.sign(y),np.log1p(y_abs))
        else:
            return np.zeros(shape=[len(self._fin_colixs)])

    def _get_aux_vector(self,cur_idx):
        if cur_idx < self._data_len:
            x = self._aux_inputs[cur_idx]
            return x
        else:
            return np.zeros(shape=[len(self._aux_colixs)])
			
    def next_batch(self):
        """
        Fetches next batch via the `_next_batch` method (if not already saved),
        saves batch onto the `_batch_cache` attribute list and also returns it.
        Also updates `_batch_cursor` to point to the following batch.
        """
        def get_next_batch(self):
            """
            Generate the next batch of sequences from the data.
            Returns:
              A batch of type Batch (see class def below)
            """
            def get_step_for_batch(self, step, seq_lengths):
                """
                Get step for entire batch.
                """
                def get_step_for_seq(self, step, batch_seq_num):
                    """
                    Get step for a particular sequence.
                    """
                    seq_x = np.zeros(self._num_inputs)
                    seq_y = np.zeros(self._num_outputs)
                    stride = self._stride
                    forecast_n = self._forecast_n
                    len1 = len(self._fin_colixs)
                    len2 = len(self._aux_colixs)
                    
                    # Get generally-relevant indices about the sequence
                    global_seq_num = self._index_cursor[batch_seq_num]
                    seq_start_idx = self._start_indices[global_seq_num]
                    seq_end_idx = self._end_indices[global_seq_num]
                    seq_lengths[batch_seq_num] = ((seq_end_idx-seq_start_idx)\
                                                  //stride)+1
                    
                    # Get indices particular to this step within the sequence
                    seq_step_idx = seq_start_idx + step*stride
                    assert(seq_step_idx < self._data_len)
                    seq_step_date = self._dates[seq_step_idx]
                    seq_step_key = self._keys[seq_step_idx]
                    nextstep_idx = seq_step_idx + forecast_n
                    if nextstep_idx < len(self._keys):
                        next_key = self._keys[nextstep_idx]
                    else:
                        next_key = ""
                        
                    # Store that step-in-the-sequence's data onto x, y, and attr
                    if seq_step_idx > seq_end_idx:
                        seq_attr = None
                        seq_x[:] = 0.0  # TODO: better way to impute this?
                        seq_y[:] = 0.0  # TODO: better way to impute this?
                    else:
                        seq_attr = (seq_step_key, seq_step_date)
                        seq_x[0:len1] = self._get_feature_vector(seq_end_idx,
                            seq_step_idx)
                        if len2 > 0:
                            seq_x[len1:len1+len2] = self._get_aux_vector(
                                seq_step_idx)
                        
                        if seq_step_key == next_key: # targets exist
                            seq_y[:] = self._get_feature_vector(seq_end_idx,
                                nextstep_idx)
                        else: # no targets exist
                            seq_y[:] = None
                    
                    return seq_x, seq_y, seq_attr
                
                # Initialize arrays and list to populate
                x = np.zeros(shape=(self._batch_size, self._num_inputs),
                        dtype=np.float)
                y = np.zeros(shape=(self._batch_size, self._num_outputs),
                        dtype=np.float)
                attr = list()
                
                # Populate arrays
                for batch_seq_num in range(self._batch_size):
                    seq_x, seq_y, seq_attr = get_step_for_seq(self, 
                        step, batch_seq_num)
                    x[batch_seq_num] = seq_x
                    y[batch_seq_num] = seq_y
                    attr.append(seq_attr)
                    
                return x, y, attr
            
            normalizers = self._get_batch_normalizers()
            seq_lengths = np.array([self._max_unrollings]*self._batch_size)
            
            inputs = list()
            targets = list()
            attribs = list()
            for step in range(self._max_unrollings):
                x, y, attr = get_step_for_batch(self, step, seq_lengths)
                inputs.append(x)
                targets.append(y)
                attribs.append(attr)
                
            assert len(inputs) == len(targets)
            
            # Set cursor for next batch
            batch_size = self._batch_size
            num_idxs = len(self._start_indices)
            self._index_cursor = [(self._index_cursor[b]+1) % num_idxs \
                                  for b in range(batch_size)]
            
            return Batch(inputs, targets, attribs, normalizers, seq_lengths)

        b = None
        # Get batch
        if self._batch_cache[self._batch_cursor] is not None:
            # if it's already cached then just get it from `_batch_cache` list
            b = self._batch_cache[self._batch_cursor]
        else:
            # otherwise get it and cache it onto `_batch_cache` list
            b = get_next_batch(self)
            self._batch_cache[self._batch_cursor] = b

        # Update `_batch_cursor` to point to next batch (wrap around if needed)
        self._batch_cursor = (self._batch_cursor+1) % (self._num_batches)

        return b

    def cache(self, verbose=False):
        """
        Caches data if not already cached.
        Does so by either reading cache from a pickled file in the _bcache
        directory, or by loading the cache (via the `_load_cache` method) and
        subsequently writing that to the _bcache directory as a pickled file
        for posterity.
        """
        def get_cache_filename(self):
            key_list = sorted(set(self._data[self._config.key_field]))
            keys = ''.join(key_list)
            sd = self._start_date if self._start_date is not None else 100001
            ed = self._end_date if self._end_date is not None else 999912
            uid = "%d-%d-%d-%d-%d-%d-%d-%s-%s-%s" % (
                    self._config.cache_id, sd, ed,
                    self._max_unrollings,
                    self._min_unrollings,
                    self._stride,
                    self._batch_size,
                    self._config.financial_fields,
                    self._config.aux_fields,
                    keys)

            hashed = hashlib.md5(uid.encode()).hexdigest()
            filename = "bcache-%s.pkl" % hashed
            return filename

        def read_cache_from_pickle(self, filepath, verbose):
            start_time = time.time()
            if verbose is True:
                print("Reading cache from %s ..." % filepath, end=' ')
            self._batch_cache = pickle.load(open(filepath, "rb"))
            self._num_batches = len(self._batch_cache)  # TODO: remove line?
            if verbose is True:
                print("done in %.2f seconds."%(time.time() - start_time))
            return

        def load_cache(self, verbose):
            """
            Caches batches from self by calling the `next_batch()` method
            (which writes batch to the list held by the `_batch_cache`
            attribute).
            """
            start_time = time.time()
            if verbose is True:
                print("\nCaching batches...", end=' '); sys.stdout.flush()

            self.rewind()
            for _ in range(self._num_batches):
                b = self.next_batch()

            if verbose is True:
                print("done in %.2f seconds." % (time.time() - start_time))

        def write_cache_to_pickle_at_filepath(self, filepath, verbose):
            start_time = time.time()
            if verbose is True:
                print("Writing cache to %s ..."%filepath, end=' ')
            pickle.dump(self._batch_cache, open(filepath, "wb"))
            if verbose is True:
                print("done in %.2f seconds."%(time.time() - start_time))

        assert len(self._batch_cache)
        if all(self._batch_cache):
            # data already cached
            pass
        else:
            # cache is (at least partially) empty (not all data already cached)
            if self._config.cache_id is not None:
                # cache WILL be read from pickle or loaded and saved onto pickle
                filename = get_cache_filename(self)
                dirname = '_bcache'
                filepath = os.path.join(dirname, filename)

                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                if os.path.isfile(filepath):
                    # read cache if already been loaded and pickled
                    read_cache_from_pickle(self, filepath)
                else:
                    # otherwise load it and pickle it for posterity
                    load_cache(self, verbose)
                    write_cache_to_pickle_at_filepath(self, filepath, verbose)
            else:
                # cache WILL NOT be read from pickle or saved as pickle
                load_cache(self, verbose)
        return

    def train_batches(self):
        """
        Returns a BatchGenerator object built from the subset of self._data that
        corresponds to the 'keys' (uniquely-identified companies) that are _not_
        in the validation set.
        """
        valid_keys = list(self._validation_set)
        valid_rows = self._data[self._config.key_field].isin(valid_keys)
        train_data = self._data[~valid_rows]
        return BatchGenerator("", self._config, validation=False,
                                  data=train_data)

    def valid_batches(self):
        """
        Returns a BatchGenerator object built from the subset of self._data that
        corresponds to the 'keys' (uniquely-identified companies) that _are_ in
        the validation set.
        """
        valid_keys = list(self._validation_set)
        valid_rows = self._data[self._config.key_field].isin(valid_keys)
        valid_data = self._data[valid_rows]
        return BatchGenerator("", self._config, validation=False,
                                  data=valid_data)

    def shuffle(self):
        if all(self._batch_cache):
            # We cannot shuffle until the entire dataset is cached
            random.shuffle(self._batch_cache)
            self._batch_cusror = 0
        return

    def rewind(self):
        """
        Resets _batch_cursor index to ensure we're working with the first batch.
        """
        self._batch_cursor = 0

    def get_scaling_params(self, scaler_class):
        if self._scaling_params is None:
            stride = self._stride
            data = self._data
            sample = list()
            z = zip(self._start_indices,self._end_indices)
            indices = random.sample(list(z),
                                    int(0.10*len(self._start_indices)))
            for start_idx, end_idx in indices:
                step = random.randrange(self._min_unrollings)
                cur_idx = start_idx+step*stride
                x1 = self._get_feature_vector(end_idx,cur_idx)
                sample.append(x1)
                #x2 = self._get_aux_vector(i,idx)
                #sample.append(np.append(x1,x2))

            scaler = None
            if hasattr(sklearn.preprocessing, scaler_class):
                scaler = getattr(sklearn.preprocessing, scaler_class)()
            else:
                raise RuntimeError("Unknown scaler = %s"%scaler_class)

            scaler.fit(sample)

            params = dict()
            params['center'] = scaler.center_ if hasattr(scaler,'center_') else scaler.mean_
            params['scale'] = scaler.scale_

            num_aux = len(self._aux_colixs)
            if num_aux > 0:
                params['center'] = np.append(params['center'], np.full( (num_aux), 0.0 ))
                params['scale'] = np.append(params['scale'], np.full( (num_aux), 1.0 ))

            self._scaling_params = params

        return self._scaling_params

    def get_raw_inputs(self,batch,idx,vec):
        len1 = len(self._fin_colixs)
        len2 = len(self._aux_colixs)
        n = batch.normalizers[idx]
        x = vec[0:len1]
        y = n * np.multiply(np.sign(x),np.expm1(np.fabs(x)))
        if len2 > 0 and len(vec) > len1:
            assert(len(vec)==len1+len2)
            y = np.append( y, vec[len1:len1+len2] )
        return y

    def get_raw_outputs(self,batch,idx,vec):
        if self._price_target_idx >= 0:
            return vec
        else:
            return self.get_raw_inputs(batch,idx,vec)

    @property
    def feature_names(self):
        return self._input_names

    @property
    def dataframe(self):
        return self._data

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def max_unrollings(self):
        return self._max_unrollings

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

class Batch(object):
    def __init__(self, inputs, targets, attribs, normalizers, seq_lengths):
        self._inputs = inputs
        self._targets = targets
        self._attribs = attribs
        self._normalizers = normalizers
        self._seq_lengths = seq_lengths

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def attribs(self):
        return self._attribs

    #@property
    #def size(self):
    #    return len(self._attribs)

    @property
    def normalizers(self):
        return self._normalizers

    @property
    def seq_lengths(self):
        return self._seq_lengths
