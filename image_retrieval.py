import os
import os.path
import json
import argparse
import logging
import requests


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path",
    dest="image_path",
    required=True,
    help="Destination folder for images and metadata (will be created if it doesn't exist)."
)
parser.add_argument(
    "--url",
    dest="url",
    default="https://collectionapi.metmuseum.org/public/collection/v1/objects",
    help="URL of the REST API to use."
)
args = parser.parse_args()

if not os.path.exists(args.image_path):
    os.makedirs(args.image_path)

logging.basicConfig(level=logging.INFO)
    
response = requests.get(
    args.url,
    params={"departmentIds" : [11]}
)

query_results = response.json()

for obj_id in sorted(query_results["objectIDs"])[0:100]:
    image_file = os.path.join(args.image_path, "{}.jpg".format(obj_id))
    metadata_file = os.path.join(args.image_path, "{}.json".format(obj_id))
    if os.path.exists(image_file):
        logging.info(
            "Skipping image with id '%s' because we already have it",
            obj_id
        )
    else:
        try:
            response = requests.get(
                os.path.join(args.url, str(obj_id)),
                timeout=3
            )
            obj = response.json()
            logging.info(
                "Processing '%s'",
                obj["title"]
            )
            if obj["primaryImage"]:
                img = requests.get(
                    obj["primaryImage"],
                    timeout=3
                )
                with open(image_file, 'wb') as ofd:
                    ofd.write(img.content)
                with open(metadata_file, 'wt') as ofd:
                    ofd.write(json.dumps(obj, indent=4))
        except requests.exceptions.ConnectTimeout() as e:
            logging.info(
                "Skipping image with id '%s' for now, because the request timed out",
                obj_id
            )
