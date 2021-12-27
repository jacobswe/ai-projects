#!/bin/bash

rm -rf data
mkdir data
cd data
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip