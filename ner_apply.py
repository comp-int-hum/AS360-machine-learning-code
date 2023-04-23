import argparse
import os.path
import logging
import warnings
import csv
import json
import re
import os.path
from glob import glob
from argparse import ArgumentParser    
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

warnings.simplefilter("ignore")

csv.field_size_limit(1000000000)

    
parser = ArgumentParser()
parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be trained on."
)
parser.add_argument(
    "--results",
    dest="results",
    required=True,
    help="The file to write the model output to"
)
parser.add_argument(
    "--model", 
    dest="model",
    default="dslim/bert-base-NER",
    help="The name of a Huggingface sequence tagging model."
)
parser.add_argument(
    "--content_field",
    dest="content_field",
    default="full_content",
    help="The CSV field containing the primary text content"
)
args = parser.parse_args()


outputs = []
processor = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForTokenClassification.from_pretrained(args.model)
ner = pipeline("ner", model=model, tokenizer=processor)
with open(args.data, "rt") as ifd:
    cifd = csv.DictReader(ifd, delimiter="\t")
    for row in cifd:
        for m in re.finditer(r"([A-Z][^\?\!\.]+[\.\?\!])\s+", row[args.content_field]):
            sentence = m.group(1)
            results = ner(sentence)
            spans = []
            current_span_start = 0
            current_span_end = 0
            current_span_label = None
            for result in results:
                et = result["entity"]
                start = result["start"]
                end = result["end"]
                if et.startswith("B"):
                    if current_span_start != current_span_end:
                        spans.append(
                            (
                                sentence[current_span_start:current_span_end],
                                current_span_label
                            )
                        )
                    
                    if start != current_span_end:
                        spans.append(
                            (
                                sentence[current_span_end:start],
                                None
                            )
                        )
                    current_span_label = et[2:]
                    current_span_start = start                    
                current_span_end = end
            spans.append(
                (
                    sentence[current_span_start:current_span_end],
                    current_span_label
                )
            )
            if current_span_end < len(sentence):
                spans.append(
                    (
                        sentence[current_span_end:],
                        None
                    )
                )
            outputs.append(
                (
                    {k : v for k, v in row.items() if k != args.content_field},
                    spans
                )
            )

        
with open(args.results, "wt") as ofd:
    ofd.write(
        json.dumps(
            outputs,
            indent=4
        )
    )
