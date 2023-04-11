import json

with open("classification_output.json", "rt") as ifd:
    output = json.loads(ifd.read())
    
for item in output:
    print(
        "Title: {}\nAuthor: {}\nPrediction: {}\n\t{}\n\n".format(
            item["title"],
            item["author"],
            item["prediction"],
            item["content"]
        )
    )
