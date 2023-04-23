import argparse
import json
import pickle
import os.path
import logging
from glob import glob
from matplotlib.figure import Figure
from matplotlib.table import Table, table
import matplotlib.pyplot as plt
from PIL import Image
import numpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path",
    dest="image_path",
    required=True,
    help="Path to previously downloaded images and metadata."
)
parser.add_argument(
    "--figure",
    dest="figure",
    required=True,
    help="Image file name to save figure to."
)
parser.add_argument(
    "--title",
    dest="title",
    help="Title for top of figure.",
    default="Change in named entity types"
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

for image_file in glob(os.path.join(args.image_path, "*jpg")):
    base, _ = os.path.splitext(image_file)
    result_file = "{}_results.json".format(base)
    with open("{}.json".format(base), "rt") as ifd:
        metadata = json.loads(ifd.read())
    logging.info(
        "Processing image '%s'.",
        metadata["title"]
    )
    image = Image.open(image_file)
    image_trans = Image.fromarray(numpy.uint8(image)).convert("RGB")
