# Bring in libraries at the very top of your code.
import csv
import re
import logging
import random
import pickle
import argparse

# You can pick out specific functions or classes if you want to refer
# to them directly in your code
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy

# You can import a (sub)module under an alias, to avoid
# having to type out a long name.
#
# Note: "module", "package", "library", "submodule"...
# these tend to be used more or less interchangeably.
# They are technically differences, but they aren't
# relevant for most situations.
import gensim.parsing.preprocessing as gpp


logging.basicConfig(level=logging.INFO)


# A silly default setting in the csv library needs to be changed 
# to handle larger fields.
csv.field_size_limit(1000000000)

# Defining arguments lets you run lots of different variants
# without editing the code, and provides clear documentation
# of what choices exist for this step.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be trained on."
)
parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="File to save the resulting model to."
)
parser.add_argument(
    "--content_field",
    dest="content_field",
    default="full_content",
    help="The CSV field containing the primary text content"
)
parser.add_argument(
    "--num_topics",
    dest="num_topics",
    default=10,
    type=int,
    help="Number of topics."
)
parser.add_argument(
    "--subdocument_length",
    dest="subdocument_length",
    default=200,
    type=int,
    help="The number of tokens to have in each subdocument."
)
parser.add_argument(
    "--minimum_word_length",
    dest="minimum_word_length",
    default=3,
    type=int,
    help="Minimum word length."
)
parser.add_argument(
    "--maximum_subdocuments",
    dest="maximum_subdocuments",
    type=int,
    help="Maximum number of subdocuments to use."
)
parser.add_argument(
    "--minimum_word_count",
    dest="minimum_word_count",
    default=5,
    type=int,
    help="Minimum number of times a word must occur in the training data to consider it 'not noise'."
)
parser.add_argument(
    "--maximum_word_proportion",
    dest="maximum_word_proportion",
    default=0.5,
    type=float,
    help="Maximum proportion of subdocuments a word can occur in before considering it 'too common'."
)
parser.add_argument(
    "--chunksize",
    dest="chunksize",
    default=2000,
    type=int,
    help="How many subdocuments to consider 'at a time': this affects how much the model 'jumps around' during training."
)
parser.add_argument(
    "--passes",
    dest="passes",
    default=20,
    type=int,
    help="How many times to 'pass over' the data during training."
)
parser.add_argument(
    "--iterations",
    dest="iterations",
    default=20,
    type=int,
    help="A highly-technical parameter for the training process (see the GenSim documentation if interested)."
)
parser.add_argument(
    "--random_seed",
    dest="random_seed",
    type=int,
    help="If this is set to a number, the training should produce the same model when given the same data and parameters."
)
args = parser.parse_args()

# Set the random number generator if the user gave an initial value.
if args.random_seed:
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

subdocuments = []

# Open your file to read ("r") in text mode ("t") as a variable 
# ("ifd" is what I use when reading files, it stands for 
# "input file descriptor").
with open(args.data, "rt") as ifd:
    
    # Use the file handle to create a CSV file handle, specifying 
    # that the delimiter is actually <TAB> rather than <COMMA>.
    cifd = csv.DictReader(ifd, delimiter="\t")
    
    # Iterate over each row of your file: since we used DictReader 
    # above, each row will be a dictionary.
    for row in cifd:
        
        # Before, we simply split the full content on whitespace.
        #
        # Here, we'll also convert to lowercase, and use a few 
        # GenSim functions to remove very short words, stopwords,
        # and non-alphanumeric characters (e.g. punctuation).
        tokens = gpp.split_on_space(
            gpp.strip_short(
                gpp.remove_stopwords(
                    gpp.strip_non_alphanum(
                        row[args.content_field].lower()
                    ),
                ),
                minsize=args.minimum_word_length,                
            )
        )

        # Calculate, based on how many tokens are in the document 
        # and the desired subdocument length, how many subdocuments 
        # you'll be creating from this document.
        num_subdocuments = int(len(tokens) / args.subdocument_length)
        
        # The subdocuments don't exist yet, but you know how many there will be, so 
        # iterate over the subdocument number (i.e. 0, 1, 2 ... /num_subdocuments/).
        # So, in this loop, you're constructing the subdocument number /subnum/.
        for subnum in range(num_subdocuments):
            
            # Each subdocument has /subdocument_length/ tokens, and you are 
            # on subdocument number /subnum/, so it will start with token 
            # number /subnum/ * /subdocument_length/.
            start_token_index = subnum * args.subdocument_length
            
            # An easy way to know where this subdocument will end: the start of 
            # the next subdocument!
            end_token_index = (subnum + 1) * args.subdocument_length
            
            # You now know where this subdocument starts and ends, so "slice" out 
            # the corresponding tokens.
            subdocument_tokens = tokens[start_token_index:end_token_index]
            
            subdocuments.append(subdocument_tokens)

# Randomly shuffle the list of subdocuments so we get a nice sampling of all the
# different material in the dataset.
random.shuffle(subdocuments)

# Only use up to /maximum_subdocuments/ for our model.
subdocuments = subdocuments[0:args.maximum_subdocuments if args.maximum_subdocuments else len(subdocuments)]

# Much like the /LabelEncoder/ and /CountVectorizer/ from last week, a GenSim 
# /Dictionary/ (not to be confused with a standard Python dictionary /{}/) will
# assign a unique integer to each word.
dct = Dictionary(documents=subdocuments)

# The /Dictionary/ also gathers total word-counts so we can filter based on how
# "interesting" they seem.  Here, we only keep words that occur at least 20 times
# in the entire dataset, and in no more than 50% of the subdocuments.
dct.filter_extremes(no_below=args.minimum_word_count, no_above=args.maximum_word_proportion)

# This unfortunate little statement is to "force" the /Dictionary/ to actually do
# its job!
temp = dct[0]

# The /Dictionary/ can now be used to transform each subdocument into word-counts
# (a row from the matrix in the lecture slides).
train = [dct.doc2bow(subdoc) for subdoc in subdocuments]

# We can now train the topic model.
model = LdaModel(
    train,
    num_topics=args.num_topics,
    id2word=dct,
    alpha="auto",
    eta="auto",
    iterations=args.iterations,
    passes=args.passes,
    eval_every=None,
    chunksize=args.chunksize
)

# Save the trained model (note the "wb" and "ofd": we're writing binary output).
with open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(model))
