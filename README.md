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
Do not use models built with this dataset for actually trading or investing.
This is a freely available dataset assembed from freely available sources and
may contain errors such as look-ahead bias and survivorship bias.

## Building Models

## Generating Forecasts

## Running the System Test

`python scripts/deep_quant.py --config=config/system_test.conf --train=True`

or, for python3:

`python3 scripts/deep_quant.py --config=config/system_test.conf --train=True`
