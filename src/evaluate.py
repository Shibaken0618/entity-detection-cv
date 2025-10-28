import os
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

from dataset import DetectionDataset, collate_fn
from model import DetectionModel
from utils import calc_iou
from custom_types import BBox


class DetectionEvaluator:
    """"
        Evaluator class to handle evaluation on validation dataset.
        1. Initialize with model path, validation dataset path, device (cpu or gpu).
        2. tp_fp_fn: Calculate true positives, false positives, false negatives for a given class.
        3. evaluate: Evaluate model on validation set, return precision, recall, F1-score per class and overall.
        4. report: Print and save evaluation report to JSON file.
        5. visualize_predictions: Visualize predictions vs ground truth for a few samples (use is optional).
    """
    def __init__(self, model_path: str, val_dir: str, device):
        self.device = device
        self.val_dir = val_dir
        self.model = DetectionModel(classes=4)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.val_dataset = DetectionDataset(base_dir=val_dir, split="val",
                                            transforms=DetectionDataset.get_transforms(train=False))
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,   # evaluate one image at a time
            shuffle=False,
            collate_fn=collate_fn
        )
        self.class_names = {0: "background", 1: "TitleBlock", 2: "Note", 3: "View"}
          

    def tp_fp_fn(self, predictions, ground_truths, class_id, iou_threshold):
        """
            Calculate true positives, false positives, false negatives for a specific class across all images.
            Uses IoU thresholding to determine matches between predicted and ground truth boxes.
        """
        tp, fp, fn = 0, 0, 0
        # iterate through each image's predictions and ground truths to calculate TP, FP, FN
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes'][pred['labels'] == class_id].cpu().numpy()
            pred_scores = pred['scores'][pred['labels'] == class_id].cpu().numpy()
            gt_boxes = gt['boxes'][gt['labels'] == class_id].cpu().numpy()
            # Bounding box (BBox) format conversion to calculate IoU
            pred_bboxes = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in pred_boxes]
            gt_bboxes = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in gt_boxes]
            gt_matches = [False] * len(gt_bboxes) # map to track matched GT boxes to predictions
            for pred_bbox in pred_bboxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt_bbox in enumerate(gt_bboxes):
                    if gt_matches[gt_idx]:
                        continue
                    iou = calc_iou(pred_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold:
                    tp += 1
                    gt_matches[best_gt_idx] = True
                else:
                    fp += 1
            fn += sum(1 for matched in gt_matches if not matched)
        return tp, fp, fn


    def evaluate(self, conf_threshold=0.5, iou_threshold=0.5):
        """
            Evaluate the model on the validation dataset.
            Returns precision, recall, F1-score per class and overall.
        """
        all_preds = []
        all_gts  =[]
        # gather all predictions and ground truths
        with torch.no_grad():
            for imgs, targets in self.val_dataloader:
                imgs = [img.to(self.device) for img in imgs]
                preds = self.model.predict(imgs, conf_threshold)
                for pred, target in zip(preds, targets):
                    all_preds.append(pred)
                    all_gts.append(target)
        # calculate metrics per class and overall
        class_metrics = {}
        overall_metrics = {"TP": 0, "FP": 0, "FN": 0} # for mAP calculation (TP = true positives, FP = false positives, FN = false negatives)
        for class_id in [1, 2, 3]:
            class_name = self.class_names[class_id]
            tp, fp, fn = self.tp_fp_fn(all_preds, all_gts, class_id, iou_threshold)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            class_metrics[class_name] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score
            }
            overall_metrics["TP"] += tp
            overall_metrics["FP"] += fp
            overall_metrics["FN"] += fn
    
        overall_precision = overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FP"]) if (overall_metrics["TP"] + overall_metrics["FP"]) > 0 else 0.0
        overall_recall = overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FN"]) if (overall_metrics["TP"] + overall_metrics["FN"]) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        overall_metrics.update({
            "Precision": overall_precision,
            "Recall": overall_recall,
            "F1-Score": overall_f1
        })

        return class_metrics, overall_metrics


    def report(self, save_path="None"):
        """
            Generate and print evaluation report.
            Saves the report to a JSON file if save_path is provided.
        """
        print("Creating evaluation report...")
        conf_thresholds = [0.3, 0.5, 0.7, 0.9]   # evaluate at multiple confidence thresholds
        results = {}
        for conf_threshold in conf_thresholds:
            class_metrics, overall_metrics = self.evaluate(conf_threshold=conf_threshold, iou_threshold=0.5)
            results[f"Conf_{conf_threshold}"] = {
                "Class_Metrics": class_metrics,
                "Overall_Metrics": overall_metrics
            }
        # print results to console
        print("\nEvaluation Results:")
        for conf_threshold, result in results.items():
            print(f"\nConfidence Threshold: {conf_threshold}")
            print("Class-wise Metrics:")
            for class_name, metrics in result["Class_Metrics"].items():
                print(f"  {class_name}: Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1-Score: {metrics['F1-Score']:.4f}")
            overall = result["Overall_Metrics"]
            print(f"Overall: Precision: {overall['Precision']:.4f}, Recall: {overall['Recall']:.4f}, F1-Score: {overall['F1-Score']:.4f}")
        # save results to JSON file
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nEvaluation report saved to {save_path}")

        return results


    def visualize_predictions(self, samples=5, conf_threshold=0.5):
        """
            Visualize predictions vs ground truth for a few samples from the validation set.
            Display images with bboxes for ground truth and predictions.
        """
        print(f"Visualizing predictions for {samples} samples...")
        colors = {1: 'r', 2: 'g', 3: 'b'}
        fig, axes = plt.subplots(samples, 2, figsize=(15, 4*samples))
        if samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(self.val_dataloader):
                if idx >= samples:
                    break
                imgs_gpu = [img.to(self.device) for img in imgs]
                preds = self.model.predict(imgs_gpu, conf_threshold)
                img = imgs[0].permute(1, 2, 0).cpu().numpy()
                target = targets[0]
                pred = preds[0]
                # Ground Truth
                ax1 = axes[idx, 0]
                ax1.imshow(img)
                ax1.set_title(f"Ground Truth (Img {idx})")
                ax1.axis('off')
                for box, label in zip(target['boxes'], target['labels']):
                    box = box.cpu().numpy()
                    label = label.cpu().item()
                    if label in colors:
                        rect = patches.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor=colors[label], facecolor='none'
                        )
                        ax1.add_patch(rect)
                        ax1.text(box[0], box[1]-5, self.class_names[label],
                                 color=colors[label], fontsize=8, weight='bold')
                # Predictions
                ax2 = axes[idx, 1]
                ax2.imshow(img)
                ax2.set_title(f"Predictions (Img {idx})")
                ax2.axis('off')
                for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                    box = box.cpu().numpy()
                    label = label.cpu().item()
                    score = score.cpu().item()
                    if label in colors:
                        rect = patches.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor=colors[label], facecolor='none'
                        )
                        ax2.add_patch(rect)
                        ax2.text(box[0], box[1]-5, f"{self.class_names[label]}: {score:.2f}",
                                 color=colors[label], fontsize=8, weight='bold')
        
        plt.tight_layout()
        plt.show()
        


def evaluation(model_path, val_dir, results_path=None):
    """
        Run evaluation on the validation dataset using the trained model.
        Saves the evaluation report to a JSON file.
        Visualizes a few predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    evaluator = DetectionEvaluator(model_path=model_path, val_dir=val_dir, device=device)
    if results_path is None:
        results_path = model_path.replace('.pth', '_evaluation_results.json')
    results = evaluator.report(save_path=results_path)
    evaluator.visualize_predictions(samples=5, conf_threshold=0.5)
    return results

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_path = config["model_output_path"]
    val_dir = config["val_path"]
    # check if model path exists, just in case
    if os.path.exists(model_path):
        evaluation(model_path=model_path, val_dir=val_dir)
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")