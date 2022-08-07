#!/bin/bash

storage_dir=/local_datasets/Libri2Mix
LibriSpeech=/data/projects/dnn_esw/users/eladco/datasets/LibriSpeech
python_path=/data/projects/swat/envs/eladco/venv_3.8/bin/python

. ./utils/parse_options.sh

$python_path local/create_local_metadata.py --librimix_dir $storage_dir

$python_path local/get_text.py \
  --libridir $LibriSpeech \
  --split test-clean \
  --outfile data/test_annotations.csv
