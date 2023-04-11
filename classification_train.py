import csv
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


csv.field_size_limit(1000000000)


with open("example_data.tsv", "rt") as ifd:
    cifd = csv.DictReader(ifd,delimiter="\t")
    docs = []
    labels = []
    titles = []
    for row in cifd:
        tokens = row["full_content"].split()
        author = row["author"]
        title = row["title"]
        sub_len = 200
        num_subdocs = int(len(tokens)/sub_len)
        for subnum in range(num_subdocs):
            start = subnum * sub_len
            end = (subnum+1) * sub_len
            sub_tokens = tokens[start:end]
            sub_document = " ".join(sub_tokens)
            docs.append(sub_document)
            titles.append(title)
            labels.append(author)

# Create label list
label_encoder = LabelEncoder()
label_encoder.fit(labels)

encoded_labels = label_encoder.transform(labels)

# split X and y into training and testing sets, by default, it splits 75% training and 25% test
#random_state=1 for reproducibility
label_title_pairs = list(zip(encoded_labels, titles))
X_train, X_test, y_train_tuples, y_test_tuples = train_test_split(
    docs,
    label_title_pairs,
    random_state=1
)
y_train = [label for label, _ in y_train_tuples]
y_test = [label for label, _ in y_test_tuples]
title_test = [title for _, title in y_test_tuples]

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

with open("classification_model.bin", "wb") as ofd:
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
