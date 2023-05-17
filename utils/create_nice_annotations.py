import argparse
import glob
import json
import os

import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_path", default="../datasets/nice", help="path to dataset.")
    return parser


def create_val_anns(data_path):
    ann_path = os.path.join(data_path, "nice-val-5k.csv")
    val_anns = pd.read_csv(ann_path)
    assert len(val_anns) == 5000

    annotations = []
    for i, item in val_anns.iterrows():
        public_id = int(item["public_id"])
        ann_dict = {
            "caption": [item["caption_gt"]],
            "category": item["category"],
            "image": f"val/{public_id}.jpg",
        }
        annotations.append(ann_dict)

    save_path = os.path.join(data_path, "nice_val.json")
    with open(save_path, "w") as f:
        json.dump(annotations, f)

    print("validation annotation is saved to {}".format(save_path))
    return


def create_test_anns(data_path):
    test_path = os.path.join(data_path, "test")
    test_images = sorted(glob.glob(os.path.join(test_path, "*.jpg")))
    assert len(test_images) == 21377

    annotations = []
    for img in test_images:
        ann_dict = {"image": f"test/{os.path.split(img)[1]}"}
        annotations.append(ann_dict)

    save_path = os.path.join(data_path, "nice_test.json")
    with open(save_path, "w") as f:
        json.dump(annotations, f)

    print("test annotation is saved to {}".format(save_path))
    return


def main():
    args = argument_parser().parse_args()
    data_path = args.data_path
    create_val_anns(data_path)
    create_test_anns(data_path)


if __name__ == "__main__":
    main()
