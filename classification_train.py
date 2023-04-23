import csv
import re
import random
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy


csv.field_size_limit(1000000000)


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
    "--target_field",
    dest="target_field",
    required=True,
    help="The CSV field that the classifier will learn to recognize."
)
parser.add_argument(
    "--subdocument_length",
    dest="subdocument_length",
    default=200,
    type=int,
    help="The number of tokens to have in each subdocument."
)
parser.add_argument(
    "--content_field",
    dest="content_field",
    default="full_content",
    help="The CSV field containing the primary text content"
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

with open(args.data, "rt") as ifd:
    cifd = csv.DictReader(ifd,delimiter="\t")
    docs = []
    labels = []
    for row in cifd:
        tokens = row[args.content_field].split()
        target = row[args.target_field]
        num_subdocs = int(len(tokens)/args.subdocument_length)
        for subnum in range(num_subdocs):
            start = subnum * args.subdocument_length
            end = (subnum+1) * args.subdocument_length
            sub_tokens = tokens[start:end]
            sub_document = " ".join(sub_tokens)
            docs.append(sub_document)
            labels.append(target)

            
# Create label list
label_encoder = LabelEncoder()
label_encoder.fit(labels)

encoded_labels = label_encoder.transform(labels)

# split X and y into training and testing sets, by default, it splits 75% training and 25% test
#random_state=1 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    docs,
    encoded_labels,
)

#instantiate the vectorizer
count_vectorizer = CountVectorizer(stop_words='english', lowercase = True, max_features = 5000)

# learn training data vocabulary, then use it to create a document-term matrix
#combine fit and transform into a single step
X_train_dtm = count_vectorizer.fit_transform(X_train)

# Transform the test data (using fitted vocabulary) into a document-term matrix
X_test_dtm = count_vectorizer.transform(X_test)

# instantiate a Multinomial Naive Bayes model.
model = MultinomialNB()

# Train the model using the train data.
model.fit(X_train_dtm, y_train)

# Apply the model to the test data.
y_pred_class = model.predict(X_test_dtm)

# Evaluate the results.
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
acc = metrics.accuracy_score(y_test, y_pred_class)
fscore = metrics.f1_score(y_test, y_pred_class, average = "macro")
print("Accuracy: {:.03}  F-Score: {:.03}".format(acc, fscore))

with open(args.model, "wb") as ofd:
    ofd.write(
        pickle.dumps(
            (
                label_encoder,
                count_vectorizer,
                model,
                confusion_matrix
            )
        )
    )
