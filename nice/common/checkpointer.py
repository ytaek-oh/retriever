# adapted from fvcore, detectron2 checkpointer
import logging
import math
import operator
import os
from collections import defaultdict

import torch
import torch.nn as nn
from termcolor import colored
from torch.nn.parallel import DataParallel, DistributedDataParallel

from iopath.common.file_io import PathManager
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

__all__ = ["Checkpointer", "PeriodicCheckpointer"]

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class Checkpointer:

    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: Any,
    ) -> None:
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables: Dict[str, Any] = {}
        for k, v in checkpointables.items():
            self.add_checkpointable(k, v)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.path_manager: PathManager = PathManager()

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        if key in self.checkpointables:
            raise KeyError(f"Key {key} already used in the Checkpointer")
        if not hasattr(checkpointable, "state_dict"):
            raise TypeError("add_checkpointable needs an object with 'state_dict()' method.")
        self.checkpointables[key] = checkpointable

    def save(self, name: str, **kwargs: Any) -> None:
        """ Dump model and checkpointables to a file. """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}

        param_grad_dic = {k: v.requires_grad for (k, v) in self.model.named_parameters()}
        state_dict = self.model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        data["model"] = state_dict
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        logging.info("Saving checkpoint to {}".format(save_file))
        # with self.path_manager.open(save_file, "wb") as f:
        torch.save(data, save_file)
        self.tag_last_checkpoint(basename)

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> Dict[str, Any]:
        pass

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd argument expected `Union[PathLike[str], str]` but
        #  got `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self) -> List[str]:
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        pass

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.
        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.
        Returns:
            same as :meth:`load`.
        """
        pass

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.
        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore

    def _load_file(self, f: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.
        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        pass

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)
            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        pass

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        pass

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        pass


class PeriodicCheckpoint:

    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
        target_iter: Optional[int] = None  # at which iteration to save the ckpt
    ) -> None:
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: List[str] = []
        self.path_manager: PathManager = checkpointer.path_manager
        self.file_prefix = file_prefix
        self.target_iter = target_iter

    def step(self, iteration: int, **kwargs: Any) -> None:
        iteration = int(iteration)
        additional_state = {"epoch": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "{}_{:02d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        # print(f"to_delete: {file_to_delete}, but did not removed intentionally.")
                        self.path_manager.rm(file_to_delete)

        if self.target_iter is not None:
            if iteration == self.target_iter:
                self.checkpointer.save(
                    f"{self.file_prefix}_target_{self.target_iter}", **additional_state
                )

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        self.checkpointer.save(name, **kwargs)


class PeriodicCheckpointer(PeriodicCheckpoint):

    def __init__(
        self, model, save_dir, period, max_to_keep=1, file_prefix="model", target_iter=None
    ):
        checkpointer = Checkpointer(model, save_dir=save_dir)
        super().__init__(
            checkpointer,
            period,
            max_to_keep=max_to_keep,
            file_prefix=file_prefix,
            target_iter=target_iter
        )

    def load(self, path):
        self.checkpointer.load(path, checkpointables=[])


class BestCheckpointer:
    """
    Checkpoints best weights based off given metric.
    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        val_metric: str,
        eval_period: int = 1,
        mode: str = "max",
        file_prefix: str = "model_best",
        config=None
    ) -> None:
        self._period = eval_period
        self._val_metric = val_metric
        assert mode in ["max", "min"]
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_epoch = None
        self.config = config

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_epoch = iteration
        return True

    def _best_checking(self, metrics_dict):
        # metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metrics_dict is None:
            return
        else:
            metric_epoch = metrics_dict["epoch"]
            latest_metric = metrics_dict.get(self._val_metric, None)

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_epoch):
                additional_state = {
                    "epoch": metric_epoch,
                    f"best_{self._val_metric}": latest_metric
                }
                if self.config is not None:
                    additional_state["config"] = self.config.to_dict()

                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                logging.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {metric_epoch} epochs"
                )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"epoch": metric_epoch, f"best_{self._val_metric}": latest_metric}
            if self.config is not None:
                additional_state["config"] = self.config.to_dict()
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            logging.info(
                f"Saved best model as latest eval score for {self._val_metric} is "
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ epoch {self.best_epoch}."
            )
            self._update_best(latest_metric, metric_epoch)
        else:
            logging.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ epoch {self.best_epoch}."
            )

    def step(self, metrics_dict):
        # based on current epoch, step
        epoch = metrics_dict.get("epoch")
        next_epoch = epoch + 1
        if (self._period > 0 and next_epoch % self._period == 0):
            self._best_checking(metrics_dict)


def _filter_reused_missing_keys(model: nn.Module, keys: List[str]) -> List[str]:
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    """
    keyset = set(keys)
    param_to_names = defaultdict(set)  # param -> names that points to it
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)
                                ) + list(module.named_buffers(recurse=False)):
            full_name = (module_prefix + "." if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        # if one name appears missing but its alias exists, then this
        # name is not considered missing
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg_per_group = sorted(k + _group_to_str(v) for k, v in groups.items())
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join([colored(x, "blue") for x in msg_per_group])
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join("  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items())
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(sorted(group)) + "}"


def _named_modules_with_dup(model: nn.Module, prefix: str = "") -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
