import warnings
import logging
import json
import os.path
from glob import glob
from argparse import ArgumentParser
from transformers import pipeline
from PIL import Image
import numpy
import torch

warnings.simplefilter("ignore")

    
parser = ArgumentParser()
parser.add_argument(
    "--image_path",
    dest="image_path",
    required=True,
    help="Path to previously downloaded images and metadata."
)
parser.add_argument(
    "--model",
    dest="model",
    default="google/owlvit-base-patch32",
    choices=["google/owlvit-base-patch32", "facebook/detr-resnet-50"],
    help="The name of the object detection Huggingface model to use."
)
parser.add_argument(
    "--tag_field",
    dest="tag_field",
    default="tags",
    help="Metadata field name containing tags with terms."
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

zero_shot = args.model == "google/owlvit-base-patch32"
detector = pipeline(model=args.model, task="zero-shot-object-detection" if zero_shot else "object-detection")

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
    tags = [l["term"] for l in metadata.get(args.tag_field, [])]
    result = detector(
        image_trans,
        tags
    ) if zero_shot else detector(image_trans)
    
        
    with open(result_file, "wt") as ofd:
        ofd.write(
            json.dumps(
                result,
                indent=4
            )
        )



# #remember, this is a function I'm providing. To import it this way, you need to have the "image_annotation.py" file in the "util" subdirectory
# from util.image_annotation import annotateImage
# from transformers import pipeline
# from PIL import Image
# import json
# import numpy as np

# #identify the owlvit model we are going to use, then create a "zero-shot-object-detection" pipeline using transformers
# checkpoint = "google/owlvit-base-patch32"
# owlvit_detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

# #open the met_tags json file were we saved the image object ID and tags
# with open("met_tags.json", "rt") as met_entries:
#     #since this json file is formatted as one json/object per line, iterate over it line by line and read each line in as a json object
#     for entry in met_entries:
#         entry = json.loads(entry)
#         #load our image using PIL and then apply a quick transformation on it (for regularization purposes)
#         image = Image.open("./met/"+entry["image"])
#         image_trans = Image.fromarray(np.uint8(image)).convert("RGB")
#         #load each of the term tags we have stored in the JSON object into a list
#         text = [l["term"] for l in entry["tags"]]
#         print(text)

#         #invoke our zero shot object detection pipeline, with the arguments being our transformed image and our list of potential text labels
#         predictions = owlvit_detector(
#             image_trans,
#             text
#         )
#         #print our predictions and use the provided function to annotate our images
#         #notice that this implies that there is a folder in this directory called "annotated_owlvit"
#         print(predictions)
#         if predictions:
#             annotateImage(image, predictions, "annotated_owlvit/"+entry["image"])
        
