import contextlib
import json
import os

from pycocotools.coco import COCO

import pandas as pd
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.captioning import CaptionTask

from pycocotools.coco import COCO  # isort:skip
from pycocoevalcap.eval import COCOEvalCap  # isort:skip


@registry.register_task("nice_captioning")
class NICECaptionTask(CaptionTask):

    def setup(self, min_len, max_len, beam_size):
        self.min_len = min_len
        self.max_len = max_len
        self.num_beams = beam_size

    def build_model(self, cfg):
        _cfg = cfg.model_cfg
        print(f"building {_cfg.arch} model with type: {_cfg.model_type}..")
        model = super().build_model(cfg)
        print()
        return model

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        res_dir = registry.get_path("result_dir")
        print(f"Saving the evaluation results to {res_dir}")
        fname = "{}_epoch{}".format(split_name, epoch)
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=res_dir,
            filename=fname,
            remove_duplicate="image_id",
        )
        self.save_to_csv(eval_result_file, fname)

        # evaluate only when using validation split
        if self.report_metric and (split_name != "test" or split_name != "nice_test"):
            metrics = self._report_metrics(eval_result_file=eval_result_file, split_name=split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def save_to_csv(self, eval_result_file, filename):
        result_path = os.path.join(os.path.dirname(eval_result_file), f"{filename}.csv")
        to_df(eval_result_file).to_csv(result_path, index=False)
        print(f"result file in csv is saved to: {result_path}.")

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        nice_gt_root = registry.get_path("nice_root")
        coco_val = caption_eval(nice_gt_root, eval_result_file, split_name)

        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = coco_val.eval["CIDEr"]
        return coco_res


def transform_nice_for_cocoeval(nice_gt_root, split_name):
    annotation_file = f"{split_name}.json"
    ann_path = os.path.join(nice_gt_root, annotation_file)
    with open(ann_path, "r") as f:
        anns = json.load(f)
    assert isinstance(anns, list)

    annotations = []
    images = []
    for i, ann in enumerate(anns):
        image_id = int(ann["image"].split("/")[-1].split(".jpg")[0])
        ann_dict = {"image_id": image_id, "caption": ann["caption"][0], "id": i}
        annotations.append(ann_dict)
        images.append({"id": image_id})

    anno_dict = {"images": images, "annotations": annotations}
    with open(os.path.join(nice_gt_root, f"{split_name}_dict.json"), "w") as f:
        json.dump(anno_dict, f)


def caption_eval(nice_gt_root, results_file, split):
    ann_file_coco = os.path.join(nice_gt_root, f"{split}_dict.json")
    if not os.path.isfile(ann_file_coco):
        transform_nice_for_cocoeval(nice_gt_root, split)

    with contextlib.redirect_stdout(None):
        coco = COCO(ann_file_coco)
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

    # print output evaluation scores
    print(coco_eval.eval)
    return coco_eval


def to_df(results_file):
    with open(results_file, "r") as f:
        result_json = json.load(f)

    result_dict = {"public_id": [], "caption": []}
    for res in result_json:
        result_dict["public_id"].append(res["image_id"])
        result_dict["caption"].append(res["caption"])

    results = pd.DataFrame.from_dict(result_dict)
    return results
