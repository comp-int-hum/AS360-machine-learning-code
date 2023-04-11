# Intro to Computational Humanities example code

This repository contains several (sequences of) Python scripts illustrating the application of simple computational and machine learning techniques to humanistic data.  Students can study and adopt them as needed.  After cloning this repository and changing into its directory on the command line, initialize a virtual environment and install the dependencies:

```
$ python3 -m venv local
$ source local/bin/activate
$ pip install -r requirements.txt
```

Then, several different sequences of scripts can be run, to train, apply, and inspect various machine learning models.  For topic modeling:

```
$ python topic_modeling_train.py
$ python topic_modeling_apply.py
$ python topic_modeling_inspect.py
```

For author classification:

```
$ python classification_train.py
$ python classification_apply.py
$ python classification_inspect.py
```

For named-entity recognition:

```
python huggingface_apply.py --input example_data.tsv
```

For object recognition:

```
python huggingface_apply.py --input example_image.jpg
```
