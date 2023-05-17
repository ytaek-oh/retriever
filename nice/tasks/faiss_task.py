import json
import os

import faiss
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.tasks.base_task import BaseTask


def _dict_to_list(input_dict):
    return [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]


@registry.register_task("blip2_faiss_construct")
class FAISSConstruct(BaseTask):

    def __init__(self, data_root, data_name, index_name, visual_dim, load_index_path=None):
        self.data_root = data_root
        self.data_name = data_name
        self.evaluate = True

        # construct faiss w/ cosine similary as distance metric. image_id as indexing key.
        if load_index_path is None:
            self.storage = faiss.IndexIDMap(
                faiss.index_factory(visual_dim, index_name, faiss.METRIC_INNER_PRODUCT)
            )
        else:
            self.storage = faiss.read_index(load_index_path)

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        data_root = run_cfg.data_root
        data_name = run_cfg.data_name
        index_name = run_cfg.index_name
        visual_dim = run_cfg.visual_dim
        load_index_path = run_cfg.get("load_index_path", None)

        return cls(
            data_root=data_root,
            data_name=data_name,
            index_name=index_name,
            visual_dim=visual_dim,
            load_index_path=load_index_path
        )

    def add_to_storage(self, embeddings, image_ids):
        embeddings = embeddings.cpu().numpy()
        image_ids = image_ids.cpu().numpy()
        self.storage.add_with_ids(embeddings, image_ids)

    def valid_step(self, model, samples):
        # vit-g features from the model initialized by BLIP-2 w/ opt2.7b after 2nd pretraining phase 
        features = model(samples, extract_feats=True, agg_mode="avg", normalize=True)
        img_ids = samples["image_id"]
        self.add_to_storage(features, img_ids)

        return [None]

    def before_evaluation(self, model, dataset, **kwargs):
        pass

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # save faiss
        save_path = os.path.join(self.data_root, "shutterstock", "knn_storage")
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.storage, os.path.join(save_path, self.data_name))
        print("FAISS index file is saved to: {}".format(save_path))
        return {"agg_metrics": 0.0}

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        pass


@registry.register_task("blip2_faiss_assign")
class FAISSAssign(BaseTask):

    def __init__(
        self, data_root, data_storage_path, discovery_topk, retrieve_k, retrieve_max_k,
        shutterstock_ann_path
    ):
        self.evaluate = True

        self.data_root = data_root
        self.storage = faiss.read_index(data_storage_path)
        print(
            "Loaded external stroage for kNN search w/ num_samples: {} and dim: {}".format(
                self.storage.ntotal, self.storage.d
            )
        )
        """
        Params:
            discovery_topk: the number of retrieved samples per query for the discovered dataset.
            retrieve_k: the actual number of retrieved samples per query, where those retrieved
                samples will be further used for caption feature extraction. Those features are
                fused with the original query input during training. This value should be geater
                than discovery_topk.
            retrieve_max_k: upper bound for the number of retrieved samples. temporary variable.
        """
        self.discovery_topk = discovery_topk
        self.k = retrieve_k
        self.max_k = retrieve_max_k
        assert self.k >= self.discovery_topk

        self.text_processor = None  # should be set when evaluation phase starts
        self.load_web_data(shutterstock_ann_path)

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        data_root = run_cfg.data_root
        data_storage_path = run_cfg.data_storage_path
        discovery_topk = run_cfg.discovery_topk
        retrieve_k = run_cfg.retrieve_k
        retrieve_max_k = run_cfg.retrieve_max_k
        shutterstock_ann_path = run_cfg.shutterstock_ann_path

        return cls(
            data_root=data_root,
            data_storage_path=data_storage_path,
            discovery_topk=discovery_topk,
            retrieve_k=retrieve_k,
            retrieve_max_k=retrieve_max_k,
            shutterstock_ann_path=shutterstock_ann_path
        )

    def load_web_data(self, ann_path):
        print("loading web data located in {}".format(ann_path))
        web_data = BaseDataset(ann_paths=[ann_path])

        # web data by image id -> caption
        img_id_to_ann = {}
        for ann in web_data.annotation:
            image = ann["image"]
            img_id = int(image.split("/")[-1].split(".jpg")[0])
            caption = ann["caption"]
            if isinstance(caption, list):
                caption = caption[0]
            assert isinstance(caption, str)
            img_id_to_ann[img_id] = {"caption": caption, "image": image}
        self.img_id_to_ann = img_id_to_ann
        self.shutterstock_id_set = set(list(img_id_to_ann.keys()))

    def retrieve_query(self, quary_emb, max_k=200):
        quary_emb = quary_emb.cpu().numpy()
        return self.storage.search(quary_emb, max_k)

    def valid_step(self, model, samples):
        assert self.text_processor is not None  # sanity check
        results = []
        assert samples

        # query: nice dataset images
        query_features = model(samples, extract_feats=True, agg_mode="avg", normalize=True)
        # ret_img_ids: retrived image ids from shutterstock web dataset

        _, ret_img_ids = self.retrieve_query(query_features, max_k=self.max_k)
        results.extend(self.filter_captions(_dict_to_list(samples), ret_img_ids))

        return results

    def _per_sample_filter_captions(self, sample, ret_img_ids):
        filtered_ret_img_ids = []

        visited_cap = set()
        if "text_input" in sample:
            # if a retrieved caption is exactly match with the gt -> exclude
            visited_cap.add(sample["text_input"])

        for ret_img_id in ret_img_ids:
            if ret_img_id == -1:
                print(f"q_img_id {sample['image_id']}: end of assignment. break.")
                break

            if ret_img_id not in self.shutterstock_id_set:
                # in case of custom shutterstock_1m annotations;
                # filter some retrieved shutterstock instance ids missing from the annotation  
                continue

            _ret_caption = self.img_id_to_ann[ret_img_id]["caption"]
            _ret_caption = _ret_caption.replace("\n", " ").lstrip()  # remove "\n"

            ret_caption = self.text_processor(_ret_caption)  # caption by ret_img_id
            if ret_caption in visited_cap:
                continue

            visited_cap.add(ret_caption)
            filtered_ret_img_ids.append(int(ret_img_id))
            if len(filtered_ret_img_ids) == self.k:
                break

        if len(filtered_ret_img_ids) != self.k:
            print(f"retrieved size: {len(filtered_ret_img_ids)}, image_id: {sample['image_id']}")

        result = {}
        result["image_id"] = int(sample["image_id"])  # query image id
        result["ret_image_ids"] = filtered_ret_img_ids

        return result

    def filter_captions(self, samples, ret_img_ids):
        results = []
        for sample, _ret_img_ids in zip(samples, ret_img_ids):
            # dict with keys: image_id, ret_image_ids
            result = self._per_sample_filter_captions(sample, _ret_img_ids)
            results.append(result)
        return results

    def before_evaluation(self, model, dataset, **kwargs):
        self.nice_anns = dataset.annotation
        self.text_processor = dataset.text_processor

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # ## nice annotation with appending retrieved ids ###
        result_dict = {}
        for res in val_result:
            result_dict[res["image_id"]] = res  # query image as a key for fast searching

        new_annotations = []
        for nice_ann in self.nice_anns:
            image_id = int(nice_ann["image"].split("/")[-1].split(".jpg")[0])
            ret_image_ids = result_dict[image_id]["ret_image_ids"]
            nice_ann["ret_image_ids"] = ret_image_ids
            new_annotations.append(nice_ann)

        if split_name == "test":
            split_name = "nice_test"  # fix the split name for test name

        save_path = os.path.join(self.data_root, "nice", f"{split_name}_ret_ids.json")
        with open(save_path, "w") as f:
            json.dump(new_annotations, f)

        print(
            "{} annotation w/ length of {} with retrieved ids is saved to {}".format(
                split_name, len(new_annotations), save_path
            )
        )

        if "nice_val_split_" in split_name:
            # skip for constructing shutterstock annotations for extracting caption features.
            # only perform those processes on full validation and test splits.
            return

        # ## shutterstock annotations for extracting caption features ###
        all_ret_ids = []
        discovery_ret_ids = []
        for res in val_result:
            all_ret_ids.extend(res["ret_image_ids"])
            discovery_ret_ids.extend(res["ret_image_ids"][:self.discovery_topk])

        unique_ret_ids = sorted(list(set(all_ret_ids)))
        anns_shutterstock = []
        for ret_id in unique_ret_ids:
            ann_shutterstock = self.img_id_to_ann[ret_id]
            anns_shutterstock.append(ann_shutterstock)

        fname = f"shutterstock_retrieval_{split_name}.json"
        save_path = os.path.join(self.data_root, "shutterstock", fname)
        with open(save_path, "w") as f:
            json.dump(anns_shutterstock, f)
        print(
            "shutterstock subset annotations w/ length of {} retrieved from {} is saved to {}".
            format(len(anns_shutterstock), split_name, save_path)
        )

        # ## dataset discovery process for retrieving from test split of nice dataset ###
        if split_name == "nice_test":
            unique_discovery_ids = list(set(discovery_ret_ids))
            anns_discovery = []
            for ret_id in unique_discovery_ids:
                ann = self.img_id_to_ann[ret_id]
                # relative path for nice dataset to shutterstock dataset
                ann["image"] = os.path.join("../shutterstock", ann["image"])
                ann["caption"] = _preprocess_caption(ann["caption"])
                anns_discovery.append(ann)

            fname = f"discovery_{split_name}_top{self.discovery_topk}.json"
            save_path = os.path.join(self.data_root, "nice", fname)
            with open(save_path, "w") as f:
                json.dump(anns_discovery, f)
            print(
                "dataset discovery with top{}, resulting in length of {} is saved to {}".format(
                    self.discovery_topk, len(anns_discovery), save_path
                )
            )
        return

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        pass


import re  # noqa  # isort:skip


def _preprocess_caption(caption):
    caption = caption.replace("\n", " ").lstrip()
    if ": " in caption:
        caption = caption.split(": ")[1].strip()
    finded = re.search("2[0-9]{3}\s-\s|2[0-9]{3};", caption)
    if finded is not None:
        caption = " ".join(caption.split(finded.group())[1:]).strip()
    return caption
