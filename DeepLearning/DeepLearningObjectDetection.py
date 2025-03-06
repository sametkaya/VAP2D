import numpy as np
import cv2
from GraphicItems.VAP_Point import VAP_Point, VAP_Point_Type
from System import FilterKernels
from detectron2.config import get_cfg
#from detectron2 import DefaultPredictor
from detectron2.data import MetadataCatalog
from pathlib import Path

from detectron2.engine import DefaultPredictor


def predict_objects_from_image(image_skel_uint8):
    """
    Predicts objects in the given preprocessed image and returns the bounding box coordinates.

    :param image_skel_uint8: Preprocessed image (binary skeletonized image).
    :return: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    pad_width = 2
    image_skel_padded = np.pad(image_skel_uint8, pad_width=pad_width, mode='constant', constant_values=0)
    # Convert grayscale to 3-channel image if necessary
    if len(image_skel_padded.shape) == 2:
        image_skel_padded = cv2.cvtColor(image_skel_padded, cv2.COLOR_GRAY2BGR)

    image_skel_padded = image_skel_padded.astype(np.uint8)
    assert image_skel_padded.shape[2] == 3, "Image should have 3 channels (BGR format)."
    assert image_skel_padded.dtype == np.uint8, "Image dtype should be np.uint8."

    # Initialize configuration
    home = Path.cwd()
    model_path = home / "detectron2" / "model" / "faster_rcnn_R_50_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(str(model_path))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 15000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 15000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 15000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 15000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    model_weights= str(home / "DeepLearning" / "Train" / "model_final.pth")
    cfg.MODEL.WEIGHTS = model_weights
    # Get class labels from metadata
    metadata = MetadataCatalog.get("my_coco_dataset")  # Ensure dataset is registered
    class_labels = metadata.thing_classes if hasattr(metadata, "thing_classes") else ["T", "B"]
    # Initialize the predictor
    predictor = DefaultPredictor(cfg)
    # Convert to float32 for processing
    image_skel_float32 = image_skel_padded.astype(np.float32)
    # Run prediction
    outputs = predictor(image_skel_float32)

    # Extract bounding boxes and class labels
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()  # Bounding box coordinates: [x_min, y_min, x_max, y_max]
    class_indices = instances.pred_classes.numpy()  # Predicted class indices

    # Format output as [(x_min, y_min, x_max, y_max, class_label), ...]
    objects = [
        (int(np.floor(x_min))-pad_width, int(np.floor(y_min))-pad_width, int(np.ceil(x_max))-pad_width, int(np.ceil(y_max))-pad_width, class_labels[class_idx])
        for (x_min, y_min, x_max, y_max), class_idx in zip(boxes, class_indices)
    ]

    return objects


def find_branch_and_tip_points(image_skel_uint8, refined_results):
    """
    Detects branch points and tip points in a skeletonized image using hit-miss transform.
    It only searches in regions near refined object centers.

    :param image_skel_uint8: Skeletonized binary image.
    :param refined_results: List of refined bounding boxes [(x_min, y_min, x_max, y_max, class_label), ...]
    :return: Tuple of lists (vap_branch_list, vap_tip_list)
    """

    # Ensure input image is binary (0 or 255)
    image_skel_uint8 = (image_skel_uint8 > 0).astype(np.uint8) * 255


    branch_points_kernels = FilterKernels.BranchPointKernels()

    tip_points_kernels = FilterKernels.TipPointKernels()

    # Pad the skeleton image to prevent border issues
    image_skel_padded = np.pad(image_skel_uint8, pad_width=1, mode='constant', constant_values=0)

    vap_branch_list = []
    vap_tip_list = []

    # Process each detected object in `refined_results`
    for (x_min, y_min, x_max, y_max, class_label) in refined_results:
        # Extract local search region
        bbox_region = image_skel_padded[y_min:y_max, x_min:x_max]

        if bbox_region.size == 0:
            continue

        if class_label == "B":  # Branch Point Detection
            bbox_region = np.pad(bbox_region, pad_width=1, mode='constant', constant_values=0)
            for kernel in branch_points_kernels:
                branch_response = cv2.morphologyEx(bbox_region, cv2.MORPH_HITMISS, kernel)
                # Find contours
                branch_objects, _ = cv2.findContours(branch_response, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(branch_objects) > 0:
                    branch_x, branch_y = branch_objects[0][0][0]  # Extract first contour point
                    refined_x = x_min + branch_x - 2
                    refined_y = y_min + branch_y - 2
                    vap_branch_list.append(VAP_Point(refined_y, refined_x, VAP_Point_Type.BRANCH))
                    break  # Stop searching after the first match

        elif class_label == "T":  # Tip Point Detection
            bbox_region = np.pad(bbox_region, pad_width=1, mode='constant', constant_values=255)
            for kernel in tip_points_kernels:
                tip_response = cv2.morphologyEx(bbox_region,  op=cv2.MORPH_HITMISS, kernel=kernel,borderType=cv2.BORDER_CONSTANT, borderValue=1)
                # Find contours
                tip_objects, _ = cv2.findContours(tip_response, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(tip_objects) > 0:
                    tip_x, tip_y = tip_objects[0][0][0]  # Extract first contour point
                    refined_x = x_min + tip_x-2
                    refined_y = y_min + tip_y-2
                    vap_tip_list.append(VAP_Point(refined_y, refined_x, VAP_Point_Type.TIP))
                    break  # Stop searching after the first match

    return vap_branch_list, vap_tip_list








