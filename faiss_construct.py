import argparse

from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.logger import setup_logger
from nice.common.utils import setup_seeds
from nice.runners.runner_base import RunnerBase
from nice.tasks import FAISSConstruct


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def main():
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = FAISSConstruct.setup_task(cfg=cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = RunnerBase(cfg=cfg, job_id=None, task=task, model=model, datasets=datasets)
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
