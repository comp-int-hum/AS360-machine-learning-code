import csv
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

csv.field_size_limit(1000000000)

with open("classification_model.bin", "rb") as ifd:
    label_encoder, count_vectorizer, model, _ = pickle.loads(ifd.read())

with open("example_data.tsv", "rt") as ifd:
    cifd = csv.DictReader(ifd,delimiter="\t")
    docs = []
    titles = []
    authors = []
    for row in cifd:
        tokens = row["full_content"].split()
        author = row["author"]
        if author == "":
            continue
        title = row["title"]
        sub_len = 50
        num_subdocs = int(len(tokens)/sub_len)
        for subnum in range(num_subdocs):
            start = subnum * sub_len
            end = (subnum+1) * sub_len
            sub_tokens = tokens[start:end]
            sub_document = " ".join(sub_tokens)
            docs.append(sub_document)
            titles.append(title)
            authors.append(author)

X = count_vectorizer.transform(docs)
predictions = label_encoder.inverse_transform(model.predict(X))

with open("classification_output.json", "wt") as ofd:
    ofd.write(
        json.dumps(
            [
                {
                    "title" : t,
                    "author" : a,
                    "prediction" : p,
                    "content" : d} for t, a, p, d in zip(
                        titles,
                        authors,
                        predictions,
                        docs
                    )
            ],
            indent=4
        )
    ) 
