# Intro to Computational Humanities example code

This repository contains several (sequences of) Python scripts illustrating the application of simple computational and machine learning techniques to humanistic data.  Students can study and adopt them as needed.  After cloning this repository and changing into its directory on the command line, initialize a virtual environment and install the dependencies:

```
$ python3 -m venv local
$ source local/bin/activate
$ pip install -r requirements.txt
```

The scripts in this repository are meant to be easy to run but also let you control various parameters and customizations.  You will want to run these *from the command line* rather than e.g. clicking the /play/ button in VSCode.  To see the details of how to run the scripts, they can be invoked with "-h", e.g.:

```
$ python topic_modeling_train.py -h
usage: topic_modeling_train.py [-h] --data DATA --model MODEL [--num_topics NUM_TOPICS]
                               [--subdocument_length SUBDOCUMENT_LENGTH] [--minimum_word_length MINIMUM_WORD_LENGTH]
                               [--maximum_subdocuments MAXIMUM_SUBDOCUMENTS]
                               [--minimum_word_count MINIMUM_WORD_COUNT]
                               [--maximum_word_proportion MAXIMUM_WORD_PROPORTION] [--chunksize CHUNKSIZE]
                               [--passes PASSES] [--iterations ITERATIONS] [--random_seed RANDOM_SEED]

options:
  -h, --help            show this help message and exit
  --data DATA           Data file model will be trained on.
  --model MODEL         File to save the resulting model to.
  --num_topics NUM_TOPICS
                        Number of topics.
  --subdocument_length SUBDOCUMENT_LENGTH
                        The number of tokens to have in each subdocument.
...
```

Note that the only required arguments are those not in square brackets, everything else has a default value.  Do you could run:

```
$ python topic_modeling_train.py --data example_data.tsv --model model.bin
```

There are Several different sequences of scripts to train, apply, and inspect results from various machine learning models.  For topic modeling:

```
$ python topic_modeling_train.py -h
$ python topic_modeling_apply.py -h
$ python topic_modeling_inspect.py -h
```

For classification:

```
$ python classification_train.py -h
$ python classification_apply.py -h
$ python classification_inspect.py -h
```

For named-entity recognition:

```
$ python ner_apply.py -h
$ python ner_inspect.py -h
```

For object recognition:

```
$ python image_retrieval.py -h
$ python object_detection_apply.py -h
$ python object_detection_inspect.py -h
```

Each sequence ultimately produces an image file.
