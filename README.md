# Deep Quant
Deep learning on company fundamental data for long-term investing

## Setting up the Environment
Clone repo, setup environment, and install requirements:

```shell 
git clone https://github.com/euclidjda/deep-quant.git
cd deep-quant
export DEEP_QUANT_ROOT=`pwd`
pip3 install -r requirements.txt
```

## Preparing the Data
```text
Do not use models built with this dataset for actually trading or investing.
This is a freely available dataset assembed from freely available sources and
may contain errors such as look-ahead bias and survivorship bias.
```

Data is passed to `deep_quant.py` as a `.dat` file, which is a space-delimited
file. The user can either run deep-quant on the full `open_dataset.dat` that's
provided, or --if, for example, the user wants to train a model on a particular
set of tickers-- on a trimmed version of `open-dataset.dat`. To obtain this
file:

```shell
python scripts/build_datfile.py
```

To trim this file to work with a reduced dataset, first create a 'ticlist' .dat
file in the `datasets` directory, following the TICKER Market format shown
below:

```text
AAPL Nasdaq
BDE Nasdaq
CALM Nasdaq
COLM Nasdaq
ESCA Nasdaq
FLXS Nasdaq
GT Nasdaq
IPAR Nasdaq
JJSF Nasdaq
LKQ Nasdaq
```

Then, run `build_datfile.py`, specifying both the path to that 'ticlist' .dat
file and the name that you'd like the produced dataset .dat file to have. For
example:

```shell
python scripts/build_datfile.py system-test-ticlist.dat system-test-dataset.dat
```


## Building Models
You can train deep quant on a neural network of a particular type and of a
particular architecture with several other hyperparameters on a particular
dataset by first defining all of these things on a config file, and then
specifying that config file as the point of reference when running
`deep_quant.py`. Consider, for example, how deep_quant is run on
`open-dataset.dat`, as specified by `config/system-test.conf`:

```shell
python scripts/deep_quant.py --config/system-test.conf --train=True
```
This will load the corresponding data and cache it in batches in a directory
called `_bcache`, and will save model checkpoints in a directory called
`chkpts-system-test` (both of these directories will be created automatically).

A couple of notes about config files:
> * The user can specify a `.dat` file to use through the `--datafile` and the
>   `data_dir` options (note that the latter is `datasets` by default).
> * `financial_fields` is a range of columns, and should be specified as a
>   string joining the first and last columns of the `.dat` file that the user
>   wants to forecast (for example: saleq_ttm-ltq_mrq).
> * `aux_fields` is similarly also a range of columns that is equivalently
>   specified. Note, however, that these fields are strictly features; they are
>   not part of what the model is trained to predict.

## Generating Forecasts
To generate forecasts for the companies in the validation set, `deep_quant.py`
must be run with the `--train` option set to False. For example:

```shell
python scripts/deep_quant.py --config=config/system-test.conf --train=False >
forecasts.txt
```

That'll produce a file called forecasts.txt with the predicted values for every
financial feature at every timestep.

## Running the System Test
`python scripts/deep_quant.py --config=config/system_test.conf --train=True`

or, for python3:

`python3 scripts/deep_quant.py --config=config/system_test.conf --train=True`
