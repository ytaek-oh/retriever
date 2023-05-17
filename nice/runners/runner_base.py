import datetime
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel as DP

from lavis.common.dist_utils import download_cached_file, is_main_process
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.datasets.base_dataset import ConcatDataset
from lavis.runners.runner_base import RunnerBase as _RunnerBase
from nice.common.checkpointer import BestCheckpointer, PeriodicCheckpointer

# un-register the `RunnerBase` class from lavis library
if "runner_base" in registry.mapping["runner_name_mapping"]:
    registry.mapping["runner_name_mapping"].pop("runner_base", None)


def _close_ret_features(dataset):
    if isinstance(dataset, ConcatDataset):
        for d in dataset.datasets:
            d.close_ret_features()
    else:
        dataset.close_ret_features()


@registry.register_runner("runner_base")
class RunnerBase(_RunnerBase):

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
        self.setup_checkpointers()  # checkpointer options
        # setup memo
        memo = cfg.run_cfg.get("memo", None)
        if memo:
            with open(os.path.join(self.output_dir, f"{memo}.txt"), "w") as f:
                f.write(memo)
        print("output_dir: {}, memo: {}".format(self.output_dir, memo))
        self.eval_epoch_after = self.config.run_cfg.get("eval_epoch_after", None)

        # load checkpoint
        load_checkpoint = cfg.run_cfg.get("load_checkpoint", None)
        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))  # nice/
        output_dir = lib_root / self.config.run_cfg.output_dir
        if self.job_id is not None:
            output_dir = output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    @property
    def lr_scheduler(self):
        # patch lr scheduler
        lr_sched = super().lr_scheduler
        if lr_sched.iters_per_epoch is None:
            lr_sched.set_iters_per_epoch(len(self.train_loader))
        return lr_sched

    def setup_checkpointers(self):
        model_no_ddp = self.unwrap_dist_model(self.model)
        max_to_keep = self.config.run_cfg.get("max_to_keep", 1)
        target_save_iter = self.config.run_cfg.get("target_save_iter", None)

        checkpointer = PeriodicCheckpointer(
            model_no_ddp, self.output_dir, 1, max_to_keep=max_to_keep, target_iter=target_save_iter
        )
        best_checkpointer = BestCheckpointer(
            checkpointer, config=self.config, val_metric="agg_metrics"
        )
        self.checkpointer = checkpointer
        self.best_checkpointer = best_checkpointer

    def close_ret_all_features(self):
        for split_name in self.datasets:
            _close_ret_features(self.datasets[split_name])

    def train(self):
        start_time = time.time()
        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)
                print()

            # evaluation phase
            if len(self.valid_splits) > 0:
                if (
                    self.eval_epoch_after is not None and cur_epoch > 0
                    and cur_epoch < self.eval_epoch_after
                ):
                    logging.info(
                        "Evaluation for current ep ({}) is skipped, it will start from {} ep.".
                        format(cur_epoch, self.eval_epoch_after)
                    )
                    continue  # do not eval until cur_epoch == self.eval_epoch_after

                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))
                    val_log = self.eval_epoch(split_name=split_name, cur_epoch=cur_epoch)
                    if val_log is not None:
                        if is_main_process():
                            assert "agg_metrics" in val_log
                            val_log["epoch"] = cur_epoch  # log current epoch

                            # checkpoint
                            self.checkpointer.step(cur_epoch, config=self.config.to_dict())
                            self.best_checkpointer.step(val_log)
                            self.log_stats(val_log, split_name)

            else:
                # saves last checkpoint, as well as the one in target epoch if specified
                self.checkpointer.step(cur_epoch, config=self.config.to_dict())
                # especially for test set run

            if self.evaluate_only:
                break

            dist.barrier()
            print()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

        logging.info("Closing all features")
        self.close_ret_all_features()
        logging.info(f"output_dir: {self.output_dir}")
        memo = self.config.run_cfg.get("memo", None)
        if memo is not None:
            logging.info(f"memo: {memo}")

    def _save_checkpoint(self, cur_epoch, is_best=False):
        raise NotImplementedError  # custom checkpointer for saving

    def _load_checkpoint(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        try:
            self.unwrap_dist_model(self.model).load_state_dict(state_dict)
        except RuntimeError:
            logging.info(
                "Key mismatch when loading checkpoint. Trying to load the model with strict=False."
            )
            _ = self.unwrap_dist_model(self.model).load_state_dict(state_dict, strict=False)
            # logging.info(msg)
            # logging.info("Missing keys {}".format(msg.missing_keys))

        if "optimizer" in checkpoint:  # selectively load optimizer
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        if not self.use_distributed:
            logging.info("Model is wrapped with DP because not self.use_distributed")
            model = DP(model)
        model.eval()

        self.task.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
