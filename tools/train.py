#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import argparse
import logging
import os
import os.path as osp
import sys
from collections import OrderedDict
import time

import cvmatrix.utils.comm as comm
from cvmatrix.utils.checkpoint import DetectionCheckpointer
from cvmatrix.utils.config import LazyConfig

from cvmatrix.engine import DefaultTrainer, default_setup, default_argument_parser, hooks, launch
from cvmatrix.evaluation import build_evaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    project_name = osp.basename(args.config_file)[:-3]
    if cfg.train.uuid is None or cfg.train.uuid == "":
        cfg.train.uuid = time.strftime("%Y%m%d-%H%M%S")

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.train.output_dir = osp.join(
        cfg.train.output_dir, project_name, cfg.train.uuid
    )

    default_setup(cfg, args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
