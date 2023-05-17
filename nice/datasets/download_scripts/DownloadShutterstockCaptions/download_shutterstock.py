"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# original source: lavis

import argparse
import io
import os
import shelve
from multiprocessing import Pool

from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

import magic  # pip install python-magic
import pandas as pd
import requests

headers = {
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}


def argument_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_root", default="../datasets", help="path to dataset.")
    return parser


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Generating parts...")
    with shelve.open("%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)) as results:

        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = (
            (index, df[i:i + chunk_size], func)
            for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks
        )
        print(
            int(len(df) / chunk_size),
            "parts.",
            chunk_size,
            "per part.",
            "Using",
            processes,
            "processes",
        )

        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return


# Unique name based on url
def _file_name(row):
    name = os.path.join(storage_dir, f"{row.image_id}.jpg")
    return name


# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row["file"])):
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
    return row


# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(
            row["url"], stream=False, timeout=5, allow_redirects=True, headers=headers
        )
        row["status"] = response.status_code
        row["headers"] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row
    if response.ok:
        row["file"] = fname
    return row


def resize_img(req):
    resize_size = 364

    image = Image.open(req).convert("RGB")
    image = image.crop((0, 0, image.width, 260))
    image = TF.resize(image, size=resize_size)
    return image


def download_image(row):
    fname = _file_name(row)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row["status"] = 200
        row["file"] = fname
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(
            row["url"], stream=False, timeout=5, allow_redirects=True, headers=headers
        )
        row["status"] = response.status_code
        # row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row

    if response.ok:
        try:
            # some sites respond with gzip transport encoding
            response.raw.decode_content = True
            img = resize_img(io.BytesIO(response.content))
            img.save(fname)

            row["mimetype"] = magic.from_file(fname, mime=True)
            row["size"] = os.stat(fname).st_size

        except Exception as e:
            #     # This is if it times out during a download or decode
            row["status"] = 408

    row["file"] = fname
    return row


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', header=0, names=["url", "image_id", "caption"])
    df["folder"] = folder
    print("Processing", len(df), " Images:")
    return df


def df_from_shelve(chunk_size, func, dataset_name):
    print("Generating Dataframe from results...")
    with shelve.open("%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df


args = argument_parser().parse_args()

storage_dir = os.path.join(args.data_root, "shutterstock", "images")
os.makedirs(storage_dir, exist_ok=True)

# number of processes in the pool can be larger than cores
num_processes = 32
# chunk_size is how many images per chunk per process - changing this resets progress when restarting
images_per_part = 100

data_name = "shutterstock"
df = open_tsv(os.path.join(args.data_root, "shutterstock", "shutterstock_filtered.csv"), data_name)
df_multiprocess(
    df=df,
    processes=num_processes,
    chunk_size=images_per_part,
    func=download_image,
    dataset_name=data_name,
)

df = df_from_shelve(chunk_size=images_per_part, func=download_image, dataset_name=data_name)

save_path = os.path.join(args.data_root, "shutterstock", "downloaded_%s_report.tsv.gz" % data_name)
df.to_csv(
    save_path,
    compression="gzip",
    sep="\t",
    header=False,
    index=False,
)
print("The download report is saved to {}.".format(save_path))
