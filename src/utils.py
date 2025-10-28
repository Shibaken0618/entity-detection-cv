from custom_types import BBox


def calc_iou(pred_box: BBox, gt_box: BBox) -> float:
    x1p, y1p, x2p, y2p = pred_box["x1"], pred_box["y1"], pred_box["x2"], pred_box["y2"]
    x1g, y1g, x2g, y2g = gt_box["x1"], gt_box["y1"], gt_box["x2"], gt_box["y2"]

    # Calculate the intersection area
    inter_x1 = max(x1p, x1g)
    inter_y1 = max(y1p, y1g)
    inter_x2 = min(x2p, x2g)
    inter_y2 = min(y2p, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = (x2p - x1p) * (y2p - y1p)
    gt_area = (x2g - x1g) * (y2g - y1g)

    if pred_area + gt_area - inter_area == 0:  # degenerate case
        return 0.0
    else:
        iou = inter_area / (pred_area + gt_area - inter_area)
    return iou
