"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from lavis.common.optims import LinearWarmupCosineLRScheduler as _LinearWarmupCosineLRScheduler
from lavis.common.optims import cosine_lr_schedule, warmup_lr_schedule
from lavis.common.registry import registry

# unregister existing BLIP2 model from registry
if "linear_warmup_cosine_lr" in registry.mapping["lr_scheduler_name_mapping"]:
    registry.mapping["lr_scheduler_name_mapping"].pop("linear_warmup_cosine_lr", None)


@registry.register_lr_scheduler("linear_warmup_cosine_lr")
class LinearWarmupCosineLRScheduler(_LinearWarmupCosineLRScheduler):

    def __init__(
        self, optimizer, max_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1, **kwargs
    ):
        super().__init__(
            optimizer,
            max_epoch,
            min_lr,
            init_lr,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr
        )
        self._iters_per_epoch = None

    @property
    def iters_per_epoch(self):
        return self._iters_per_epoch

    def set_iters_per_epoch(self, steps):
        self._iters_per_epoch = steps

    def step(self, cur_epoch, cur_step):
        # apply warmup steps after epoch 0
        assert self.iters_per_epoch is not None
        cur_accum_step = cur_epoch * self.iters_per_epoch + cur_step
        if self.warmup_steps > 0 and cur_accum_step + 1 == self.warmup_steps:
            print("Warmup phase has ended at total accumulated {} steps".format(cur_accum_step + 1))

        if cur_accum_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_accum_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )
