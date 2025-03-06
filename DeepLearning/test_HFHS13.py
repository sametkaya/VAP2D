# import random
# from collections import defaultdict
#
# import numpy as np
# import torch
# import os
# import cv2
#
# from System.FileOperatıons import print_to_csv
# from System.PostProcessing import remove_intersect_object_from_instances, print_infos
# from detectron2.config import get_cfg
# from detectron2.data.datasets import load_coco_json, register_coco_instances
# from detectron2.engine import DefaultPredictor, DefaultTrainer
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.modeling import detector_postprocess
# from detectron2.structures import pairwise_iou, Boxes, Instances
# from detectron2.utils.visualizer import Visualizer, ColorMode
#
# data_folder = r"/home/odesa/PycharmProjects/detection3/images/data"
# data_annotations_file = os.path.join(data_folder, "coco_test_HFHS13_r_dataset_5x5.json")
# data_test_images_folder = os.path.join(data_folder, "test/HFHS13_r")
# data_test_result_folder =  os.path.join(r"/home/odesa/PycharmProjects/detection3/images/", "results/result2/test/HFHS13")
# data_train_result_folder =  os.path.join(r"/home/odesa/PycharmProjects/detection3/images/", "results/result2/train/HFHS13")
# #data_train_result_folder =  os.path.join(r"/home/odesa/PycharmProjects/detection3/images/", "results/result1/train/NC15")
# data_test_result_images_folder =  os.path.join(data_test_result_folder, "result_images")
#
#
# register_coco_instances("my_coco_dataset_test",{},data_annotations_file,data_test_images_folder)
#
# # Register the COCO dataset with Detectron2
# DatasetCatalog.register("my_coco_dataset", lambda: load_coco_json(data_annotations_file,data_test_images_folder))
#
# meta_data_catalog=MetadataCatalog.get("my_coco_dataset")
# meta_data_catalog.thing_classes=["T", "B"] # Replace with your actual class names
# meta_data_catalog.thing_colors =[(0, 255, 0), (0, 0, 255)]
#
# dataset_catalog_dic = DatasetCatalog.get("my_coco_dataset")
#
# #show_sample(dataset_catalog_dic,meta_data_catalog,None) #show new_data_raw
#
# cfg = get_cfg()
# cfg.merge_from_file(os.path.join("/home/odesa/PycharmProjects/detection3/detectron2/configs/COCO-Detection","faster_rcnn_R_50_FPN_1x.yaml"))  # You can choose other models as well
# #cfg.DATASETS.TRAIN = ("my_coco_dataset",)
# cfg.DATASETS.TEST = ("my_coco_dataset",)
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.TEST.DETECTIONS_PER_IMAGE = 10000
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 15000
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 15000
# cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 15000
# cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 15000
#
# cfg.SOLVER.BASE_LR = 0.0001 # pick a good LR
# cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16  # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#
#
# # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# cfg.OUTPUT_DIR = data_train_result_folder
# # Inference should use the config with parameters that are used in training
# # cfg now already contains everything we've set previously. We changed it a little bit for inference:
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set a custom testing threshold
# cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
# predictor = DefaultPredictor(cfg)
#
# #infos,filtered_outputs=print_infos(dataset_catalog_dic, meta_data_catalog, predictor) #show predict
#
# infos, filtered_outputs=print_infos(dataset_catalog_dic, meta_data_catalog, predictor,save_image=True,save_folder=data_test_result_images_folder) #show predict
#
# print_to_csv(data_test_result_folder,"HFHS13_faster_rcnn_R_50_FPN_1x",infos )
#
#
# print("*****raw result*****")
# evaluator_raw = COCOEvaluator("my_coco_dataset_test", cfg, False, output_dir=data_test_result_folder, max_dets_per_image=2000)
# val_loader = build_detection_test_loader(cfg, "my_coco_dataset_test")
# inference_on_dataset(predictor.model, val_loader, evaluator_raw)
#
# print("*****filtered result*****")
# evaluator_filtered = COCOEvaluator("my_coco_dataset_test", ("bbox",), False, output_dir=data_test_result_folder, max_dets_per_image=2000)
# evaluator_filtered.reset()
# data_loader = DatasetCatalog.get("my_coco_dataset_test")
# evaluator_filtered.process(data_loader, filtered_outputs)
# evaluation_results = evaluator_filtered.evaluate()
# print(evaluation_results)
# cv2.waitKey(0)
#
#
#
#
