# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import copy
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from home_robot.core.interfaces import Observations
from detectron2.structures.instances import Instances

from maskdino import add_maskdino_config
from .coco_categories import coco_categories, coco_categories_mapping


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


class COCOMaskDINO:
    def __init__(
        self,
        vocabulary: str = "coco",
        sem_pred_prob_thr: float = 0.5,
        sem_gpu_id: int = 0,
        min_depth: float = 0.5,
    ):
        """
        Arguments:
            vocabulary: currently one of "coco" for indoor coco categories or "coco-subset"
             for 6 coco goal categories
            sem_pred_prob_thr: prediction threshold
            sem_gpu_id: prediction GPU id (-1 for CPU)
        """
        parent_path = Path(__file__).resolve().parent
        config_path = f"{parent_path}/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
        ckpt_path = f"{parent_path}/checkpoints/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"

        self.visualize = True
        if vocabulary == "coco":
            self.vocabulary = coco_categories
            self.vocabulary_mapping = coco_categories_mapping
            self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        elif vocabulary == "coco-subset":
            self.vocabulary = {
                "chair": 0,
                "couch": 1,
                "plant": 2,
                "bed": 3,
                "toilet": 4,
                "tv": 5,
                "no-category": 6,
            }
            self.vocabulary_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
            }
            self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        else:
            raise ValueError("Vocabulary {} does not exist".format(vocabulary))
        self.segmentation_model = ImageSegmentation(
            config_path, ckpt_path, sem_pred_prob_thr, sem_gpu_id,
            classes_to_visualize=list(coco_categories_mapping.keys())
        )
        self.sem_pred_prob_thr = sem_pred_prob_thr
        self.num_sem_categories = len(self.vocabulary)
        self.min_depth = min_depth

    def get_prediction(
        self, images: np.ndarray, depths: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arguments:
            images: images of shape (batch_size, H, W, 3) (in BGR order)
            depths: depth frames of shape (batch_size, H, W)

        Returns:
            one_hot_predictions: one hot segmentation predictions of shape
             (batch_size, H, W, num_sem_categories)
            visualizations: prediction visualization images
             shape (batch_size, H, W, 3) if self.visualize=True, else
             original images
        """
        batch_size, height, width, _ = images.shape

        predictions, visualizations = self.segmentation_model.get_predictions(
            images
        )
        one_hot_predictions = np.zeros(
            (batch_size, height, width, self.num_sem_categories)
        )

        for i in range(batch_size):
            print("=" * 20)
            for j, class_idx in enumerate(
                predictions[i]["instances"].pred_classes.cpu().numpy()
            ):
                if class_idx in list(self.vocabulary_mapping.keys()):
                    idx = self.vocabulary_mapping[class_idx]
                    obj_mask = predictions[i]["instances"].pred_masks[j] * 1.0
                    obj_mask = obj_mask.cpu().numpy()
                    score = predictions[i]["instances"].scores[j].item()
                    if score < self.sem_pred_prob_thr:
                        continue
                    if depths is not None:
                        # Note: depth is in meters
                        depth = depths[i]
                        md = np.median(depth[obj_mask == 1])
                        if md == 0:
                            filter_mask = np.ones_like(obj_mask, dtype=bool)
                        else:
                            # Restrict objects to 1m depth
                            filter_mask = (depth >= md + 0.5) | (depth <= md - 0.5)
                            # Remove pixels within min_depth range
                            filter_mask = filter_mask | (depth <= self.min_depth)
                        obj_mask[filter_mask] = 0.0

                    one_hot_predictions[i, :, :, idx] += obj_mask
                    print("===> Found class {}".format(self.inv_vocabulary[idx]))
            print("=" * 20)

        if self.visualize:
            visualizations = np.stack([vis.get_image() for vis in visualizations])
        else:
            # Convert BGR to RGB for visualization
            visualizations = images[:, :, :, ::-1]

        return one_hot_predictions, visualizations

    def predict(
        self, obs: Observations,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in BGR order)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W, N) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        images = obs.rgb[np.newaxis, :, :, :]
        depths = obs.depth[np.newaxis, :, :]
        pred, vis = self.get_prediction(images, depths)
        pred, vis = pred[0], vis[0]
        obs.semantic = pred
        obs.task_observations["semantic_frame"] = vis
        return obs  


class ImageSegmentation:
    def __init__(
        self,
        config_path,
        ckpt_path,
        sem_pred_prob_thr,
        sem_gpu_id,
        classes_to_visualize=None,
    ):
        string_args = f"""
            --config-file {config_path}
            --confidence-threshold {sem_pred_prob_thr}
            --opts MODEL.WEIGHTS {ckpt_path}
            """

        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(
            cfg,
            classes_to_visualize=classes_to_visualize,
            sem_pred_prob_thr=sem_pred_prob_thr,
        )

    def get_predictions(self, images):
        return self.demo.run_on_images(images)


def get_seg_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, classes_to_visualize=None, sem_pred_prob_thr=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        print("\n" + "*" * 50)
        print(self.metadata)
        print("*" * 50 + "\n")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.classes_to_visualize = classes_to_visualize
        self.sem_pred_prob_thr = sem_pred_prob_thr
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                if self.classes_to_visualize is not None or self.sem_pred_prob_thr is not None:
                    n_insts = instances.scores.shape[0]
                    masks = torch.ones(n_insts).bool() # Include all instance by default
                    for i in range(n_insts):
                        if self.classes_to_visualize is not None:
                            # Remove instances outside the classes required
                            cls_i = int(instances.pred_classes[i].item())
                            if cls_i not in self.classes_to_visualize:
                                masks[i] = False
                        if self.sem_pred_prob_thr is not None:
                            # Remove instances where prediction confidence is low
                            score_i = instances.scores[i].item()
                            if score_i < self.sem_pred_prob_thr:
                                masks[i] = False
                    instances_filt = {}
                    if instances.has("pred_boxes"):
                        instances_filt["pred_boxes"] = instances.pred_boxes[masks]
                    instances_filt["scores"] = instances.scores[masks]
                    instances_filt["pred_classes"] = instances.pred_classes[masks]
                    if instances.has("pred_keypoints"):
                        instances_filt["pred_keypoints"] = instances.pred_keypoints[masks]
                    if instances.has("pred_masks"):
                        instances_filt["pred_masks"] = instances.pred_masks[masks]
                    instances_filt = Instances(
                        instances.image_size, **instances_filt
                    )
                    instances = instances_filt
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def run_on_images(self, images):
        predictions, vis_outputs = [], []
        for image in images:
            p, v = self.run_on_image(image)
            predictions.append(p)
            vis_outputs.append(v)
        return predictions, vis_outputs
    