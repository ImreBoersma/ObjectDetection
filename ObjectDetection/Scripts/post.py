import torch
from retinanet.box import generate_anchors, decode, nms

import sys
input_image = sys.argv[1]
cls_heads = sys.argv[2]
box_heads = sys.argv[3]


def detection_postprocess(image, cls_heads, box_heads):
    # Inference post-processing
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.shape[-1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        decoded.append(decode(cls_head, box_head, stride,
                              threshold=0.05, top_n=1000, anchors=anchors[stride]))

    # Perform non-maximum suppression
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    # NMS threshold = 0.5
    scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
    return scores, boxes, labels


scores, boxes, labels = detection_postprocess(
    input_image, cls_heads, box_heads)
