import argparse
import json
import os

import h5py


def argument_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_root", default="../datasets", help="path to dataset.")
    # paths for first annotation and corresponding precomputed caption feature
    parser.add_argument("--anns_first", required=True, type=str)
    parser.add_argument("--feats_first", default=None, type=str)
    # paths for second annotation and corresponding precomputed caption feature
    parser.add_argument("--anns_second", required=True, type=str)
    parser.add_argument("--feats_second", default=None, type=str)
    # filenames for the concatenated annotation and corresponding caption feature
    parser.add_argument("--anns_save_name", required=True, type=str)
    parser.add_argument("--feats_save_name", default=None, type=str)
    return parser


def main(args):
    anns1 = json.load(open(args.anns_first, "r"))
    assert isinstance(anns1, list)
    anns2 = json.load(open(args.anns_second, "r"))
    assert isinstance(anns2, list)
    anns = anns1 + anns2

    # annotation is considered as nice data
    ann_save_path = os.path.join(args.data_root, "nice", args.anns_save_name)
    with open(ann_save_path, "w") as f:
        json.dump(anns, f)
    print("resulting dataset with length {} is saved to {}".format(len(anns), ann_save_path))
    print()

    if not args.concat_feats:
        return
    concat_h5py_features(args)
    print()

    return


def concat_h5py_features(args):
    data_root = os.path.join(args.data_root, "shutterstock", "caption_features")
    feats1 = h5py.File(args.feats_first, "r")
    feats2 = h5py.File(args.feats_second, "r")

    keys1 = list(feats1.keys())
    keys2 = list(feats2.keys())

    total_progress = len(set(keys1 + keys2))
    num_steps = total_progress // 20

    # empty dataset
    feats_save_path = os.path.join(data_root, args.feats_save_name)
    h = h5py.File(feats_save_path, "w")
    visited = set()

    # copy feats1 to h
    for i, k in enumerate(keys1):
        if (i + 1) % num_steps == 0:
            print(f"{i+1}/{total_progress}")
        visited.add(k)
        h.create_dataset(k, data=feats1[k][:])

    # copy feats2 to h while removing duplicates
    for j, k in enumerate(keys2):
        if (i + 1) % num_steps == 0:
            print(f"{i+1}/{total_progress}")
        if k in visited:
            continue
        h.create_dataset(k, data=feats2[k][:])
        i += 1
    h.close()

    print(
        "independent sum of length of keys: {}+{}={}".format(
            len(keys1), len(keys2), len(keys1 + keys2)
        )
    )
    print(
        "Resulting size of features: {}, which is saved to {}".format(
            total_progress, feats_save_path
        )
    )


if __name__ == "__main__":
    args = argument_parser().parse_args()
    assert args.anns_first != args.anns_second
    if args.feats_first is not None:
        assert args.feats_second is not None
        assert args.feats_save_name is not None
    args.concat_feats = args.feats_first is not None
    main(args)
