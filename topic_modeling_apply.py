# Bring in libraries at the very top of your code.
import csv
import re
import logging
import random
import pickle
import json
import argparse

# You can pick out specific functions or classes if you want to refer
# to them directly in your code
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# You can import a (sub)module under an alias, to avoid
# having to type out a long name.
#
# Note: "module", "package", "library", "submodule"...
# these tend to be used more or less interchangeably.
# They are technically differences, but they aren't
# relevant for most situations.
import gensim.parsing.preprocessing as gpp


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="Model file generated by training script."
)
parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be applied to."
)
parser.add_argument(
    "--counts",
    dest="counts",
    required=True,
    help="Counts output file."
)
parser.add_argument(
    "--content_field",
    dest="content_field",
    default="full_content",
    help="The CSV field containing the primary text content"
)
parser.add_argument(
    "--numeric_group_field",
    dest="numeric_group_field",
    default="rights_date_used",
    help="A numeric field in the CSV to group the topic counts by."
)
parser.add_argument(
    "--categorical_group_field",
    dest="categorical_group_field",
    help="A non-numeric field in the CSV to group the topic counts by (will be used instead of numeric_group_field)."
)
parser.add_argument(
    "--group_min",
    dest="group_min",
    default=1500,
    type=int,
    help="The minimum value of the group field (any documents lower than this will be discarded)."
)
parser.add_argument(
    "--group_resolution",
    dest="group_resolution",    
    default=10,
    type=int,
    help="The size of each group (e.g. number of years, or whatever units 'group_field' uses)."
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
    help="Minimum word length"
)
args = parser.parse_args()

# A silly default setting in the csv library needs to be changed 
# to handle larger fields.
csv.field_size_limit(1000000000)

# For each "group", we'll collect the number of times each topic
# occurs.
groupwise_topic_counts = {}

group_names = {}

# Read in the model that was previously trained and serialized.
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())

# Open your file to read ("r") in text mode ("t") as a variable 
# ("ifd" is what Tom uses when reading files, it stands for 
# "input file descriptor").
with open(args.data, "rt") as ifd:
    
    # Use the file handle to create a CSV file handle, specifying 
    # that the delimiter is actually <TAB> rather than <COMMA>.
    cifd = csv.DictReader(ifd, delimiter="\t")
    
    # Iterate over each row of your file: since we used DictReader 
    # above, each row will be a dictionary.
    for row in cifd:
        
        # This =is where we decide which "group" this document belongs to,
        # either the value of the categorical field (if specified), or
        # by rounding the numeric field to the nearest range.
        if args.categorical_group_field:
            group = row[args.categorical_group_field]
        else:
            group_value = int(row[args.numeric_group_field])
            if group_value < args.group_min:
                continue
            group = group_value - (group_value % args.group_resolution)
        
        # Make sure there is a bucket for the group.
        groupwise_topic_counts[group] = groupwise_topic_counts.get(group, {})
        
        # We want to prepare the data the same way we prepared the data that
        # trained the model (there may be situations where we'd do something
        # different, but only with a particularly good reason!).
        tokens = gpp.split_on_space(
            gpp.strip_short(
                gpp.remove_stopwords(
                    gpp.strip_non_alphanum(
                        row[args.content_field].lower()
                    ),
                ),
            minsize=args.minimum_word_length
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
            
            # Turn the subdocument tokens into integers and count them, using the
            # trained model (so it employs the same mapping as it was trained with).
            subdocument_bow = model.id2word.doc2bow(subdocument_tokens)
            
            # It will be useful to have the "bag-of-words" counts as a dictionary, too.
            subdocument_bow_lookup = dict(subdocument_bow)
            
            # Apply the model to the subdocument, asking it to give the specific
            # assignments, i.e. which topic is responsible for each unique word.
            #
            # Note how an underscore ("_") can be used, here and in other situations,
            # when you *don't* want to assign something to a variable, because you
            # aren't going to need it.  This makes the code (and your intentions)
            # clear.
            _, labeled_subdocument, _ = model.get_document_topics(
                subdocument_bow,
                per_word_topics=True
            )
            
            # Add the topic counts for this subdocument to the appropriate group.
            for word_id, topics in labeled_subdocument:
                # Gensim insists on returning *lists* of topics in descending order of likelihood.  When
                # the list is empty, it means this word wasn't seen during training (I think!), so we skip
                # it.
                if len(topics) > 0:
                    
                    # Assume the likeliest topic.
                    topic = topics[0]
                    
                    # Add the number of times this word appeared in the subdocument to the topic's count for the group.
                    groupwise_topic_counts[group][topic] = groupwise_topic_counts[group].get(topic, 0) + subdocument_bow_lookup[word_id]

# Save the counts to a file in the "JSON" format.  The 'indent=4' argument makes it a lot easier
# for a human to read the resulting file directly.
with open(args.counts, "wt") as ofd:
    ofd.write(
        json.dumps(
            [(k, v) for k, v in sorted(groupwise_topic_counts.items()) if len(v) > 0],
            indent=4
        )
    )
