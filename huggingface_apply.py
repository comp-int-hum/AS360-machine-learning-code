import os.path
import logging
import warnings
import csv
import json
import re
import os.path
from glob import glob

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForObjectDetection, AutoImageProcessor
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

import torch

warnings.simplefilter("ignore")

csv.field_size_limit(1000000000)

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument(
        "--model", 
        dest="model", 
        help="The name of a Huggingface model, such as 'facebook/detr-resnet-50' for images or 'dslim/bert-base-NER' for text"
    )
    parser.add_argument(
        "--input",
        dest="input",
        default="example_data.tsv",
        help="What to apply the model to: either a CSV file, or a directory of images"
    )
    parser.add_argument(
        "--output",
        dest="output",
        default="output.json",
        help="The output file to write the model output to"
    )
    args = parser.parse_args()
    
    datatype = "documents" if args.input.endswith("tsv") else "images"
    if not args.model:
        args.model = "facebook/detr-resnet-50" if datatype == "images" else "dslim/bert-base-NER"
    
    if datatype == "documents":
        processor = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForTokenClassification.from_pretrained(args.model)
        ner = pipeline("ner", model=model, tokenizer=processor)
        with open(args.input, "rt") as ifd:
            cifd = csv.DictReader(ifd, delimiter="\t")
            for row in cifd:
                for m in re.finditer(r"([A-Z][^\?\!\.]+[\.\?\!])\s+", row["full_content"]):
                    sentence = m.group(1)
                    entities = ner(sentence)
                    if len(entities) > 0:
                        print("In sentence '{}' I found entities:".format(sentence))
                        for entity in entities:
                            print("\tType = {}, Word = {}".format(entity["entity"], entity["word"]))
    else:
        processor = DetrImageProcessor.from_pretrained(args.model)
        model = DetrForObjectDetection.from_pretrained(args.model)
        image = Image.open(args.input)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        print(10)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
