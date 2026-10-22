import numpy as np
import json, os, math, glob
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
from pprint import pprint
from typing import Union, List, Tuple, Literal
from scipy.ndimage import label, binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt 
from TnT.evaluation.mapping import location_mapping
from TnT.evaluation.plotting import heatmap, spider, decluttered_spider
from TnT.evaluation.metrics import hd95_single_label, dice_coefficient_single_label, vs_single_label

N_CLASSES=52


def extract_labels(array1: np.array, array2 = None) -> list[int]:
    """
    Extracts unique sorted labels from array input(s)
    if two arrays (such as gt and pred arrays) are input
    extract union of labels in gt and pred masks

    WITHOUT background 0
    """
    if array2 is not None:
        # numpy.union1d
        # Will be flattened if not already 1D
        labels = np.union1d(array1, array2)
    else:
        labels = np.unique(array1)
    # print("labels = ", labels)

    # remove background 0
    filtered_labels = [int(x) for x in labels[labels != 0]]
    # print(f"filtered_labels = {filtered_labels}")
    return filtered_labels

def evaluation_function(pred: sitk.Image, gt: sitk.Image, execute_in_docker=True):

    if gt.GetSize() != pred.GetSize():
        raise ValueError(
            f"GT and prediction have different sizes: "
            f"{gt.GetSize()} vs {pred.GetSize()}"
        )

    result = {
        "gt_filename": "N/A",
    }

    gt_locs = extract_labels(sitk.GetArrayFromImage(gt))
    pred_locs = extract_labels(sitk.GetArrayFromImage(pred))

    for cls in range(1, N_CLASSES + 1):
        # Detection based on segmentation

        # same as eval/task1/evaluate.py's evaluation_function
        result[f"TP_{cls}"] = int(cls in pred_locs and cls in gt_locs)
        result[f"TN_{cls}"] = int(cls not in pred_locs and cls not in gt_locs)
        result[f"FP_{cls}"] = int(cls in pred_locs and cls not in gt_locs)
        result[f"FN_{cls}"] = int(cls not in pred_locs and cls in gt_locs)

        # also report support
        result[f"support_{cls}"] = int(cls in gt_locs)

        # Segmentation metrics are undefined for true negatives.
        if result[f"TN_{cls}"]:
            result[f"DICE_{cls}"] = np.nan
            result[f"HD95_{cls}"] = np.nan
            result[f"VOLSIM_{cls}"] = np.nan
        else:
            result[f"DICE_{cls}"] = dice_coefficient_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )
            result[f"HD95_{cls}"] = hd95_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )[0]/290  # [hd95_score, hd100_score]
            result[f"VOLSIM_{cls}"] = vs_single_label(
                gt=gt,
                pred=pred,
                label=cls,
            )

    return result


def evaluation_aggregation(results: list):
    """Aggregate evaluation results by class across images.

    Detection counts (TP, TN, FP, FN) are summed across images.
    Then, precision, recall, and MCC are computed from the aggregated
    counts for each class (division-by-zero returns NaN).

    Segmentation metrics (DICE, HD95, VOLSIM) are averaged across
    images, ignoring NaN values.
    """
    aggregates = {}
    keys = results[0].keys()
    # eps = 1e-6

    # Aggregate based on detection counts vs segmentation metrics
    for k in keys:
        if k == "gt_filename":
            continue

        # all values for this class across images
        values = [result[k] for result in results]

        if k.startswith(("TP_", "TN_", "FP_", "FN_", "support_")):
            # Pool detection counts across images.
            aggregates[k] = sum(values)

        elif k.startswith(("DICE_", "HD95_", "VOLSIM_")):
            # Average over valid seg-metric values for each class.
            valid_values = [v for v in values if not np.isnan(v)]
            aggregates[k] = np.mean(valid_values) if valid_values else np.nan
            
    # for i in range(1, N_CLASSES+1):
    #     aggregates[f'is_present_{i}']=any([aggregates[f'{v}_{i}']!=0 for v in ['TP', 'FP', 'FN']])

    # Obtain detection metrics from aggregated counts
    for i in range(1, N_CLASSES + 1):
        tp = aggregates[f"TP_{i}"]
        fp = aggregates[f"FP_{i}"]
        fn = aggregates[f"FN_{i}"]
        tn = aggregates[f"TN_{i}"]
        # precision = tp/(tp+fp)
        aggregates[f"PRECISION_{i}"] = tp / (tp + fp) if tp + fp else np.nan
        # recall = tp/(tp+fn)
        aggregates[f"RECALL_{i}"] = tp / (tp + fn) if tp + fn else np.nan
        # f1 = 2 * tp / ((2 * tp) + fp + fn)
        aggregates[f"F1_{i}"] = (
            2 * tp / ((2 * tp) + fp + fn) if (tp + fp + fn) else np.nan
        )
        # mcc = (tp*tn - fp*fn)/sqrt(...)
        mcc_num = tp * tn - fn * fp
        mcc_den = math.sqrt(
            (aggregates[f"TP_{i}"] + aggregates[f"FP_{i}"])
            * (aggregates[f"TP_{i}"] + aggregates[f"FN_{i}"])
            * (aggregates[f"TN_{i}"] + aggregates[f"FP_{i}"])
            * (aggregates[f"TN_{i}"] + aggregates[f"FN_{i}"])
        )
        aggregates[f"MCC_{i}"] = mcc_num / mcc_den if mcc_den else np.nan

    # aggregates with detection and segmentation metrics for each class
    return aggregates


def nanmean(aggregates, metric_name) -> tuple[float, int]:
    """
    Return the mean across valid class values and the number of valid classes.

    NOTE: If the nanmean of a metric is still nan, ie no valid values,
    for GC leaderboard display, convert the averaged nanmean from nan to 0
    """
    values = np.asarray(
        [aggregates[f"{metric_name}_{i}"] for i in range(1, N_CLASSES + 1)]
    )

    #print(f"{metric_name} values = {values}")

    valid_values = [v for v in values if not np.isnan(v)]

    if not valid_values:
        print(f"[WARNING] {metric_name} contains all NaN")
        return 0, 0

    return float(np.mean(valid_values)), len(valid_values)


def evaluation_average(aggregates):
    """
    Compute leaderboard averages and track valid class counts.

    The average reported here is across classes:
        Across images -> evaluation_aggregation()
        Across classes -> evaluation_average()
    e.g.
        DICE_i = average Dice for class i across valid images.
        DICE = average of DICE_i across the 52 classes.

    If the nanmean of a metric after averaging is still nan,
    for GC leaderboard display, convert the averaged nanmean to 0
    """

    cls_avg = {}

    for metric in ["PRECISION", "RECALL", "MCC", "F1", "DICE", "HD95", "VOLSIM"]:
        mean, count = nanmean(aggregates, metric)
        cls_avg[metric] = mean
        cls_avg[f"count_valid_{metric}"] = count

    return cls_avg

class TopAneu26LikeEvaluator():
    def __init__(self, pipeline, wdir, precomp_predictions=None):
        self.pipeline = pipeline
        self.wdir = Path(wdir) if wdir is not None else None
        self.use_perf = precomp_predictions is None
        self.precomp = Path(precomp_predictions) if precomp_predictions is not None else None
        
    def eval_list(self, dir, files):
        mapping = {v:k for k, v in location_mapping['labels'].items()}
        results = []
        discrepancies = {}
        dir = Path(dir)
        for f in tqdm(files, desc='Evaluating'):
            img = sitk.ReadImage(dir/'images'/f)
            pred = self.pipeline(img, 'MRA' if '_mr_' in f else 'CTA')
            ref = sitk.ReadImage(dir/'location_masks'/f.replace('_0000', ''))
            res = evaluation_function(pred, ref)
            res['modality'] = 'MRA' if '_mr_' in f else 'CTA'
            results.append(res)
            
            ref_arr = sitk.GetArrayFromImage(ref)
            pred_arr = sitk.GetArrayFromImage(pred)
            cc, n = label(ref_arr, generate_binary_structure(3,3))
            for i in range(1, n+1):
                slc = cc==i
                gt_v=np.median(ref_arr[slc])
                pred_v=np.median(pred_arr[slc])
                if gt_v!=pred_v:
                    discrepancies[f.split('.')[0]+f'_{i}']={'GT':mapping[gt_v],'Predicted':mapping[pred_v]}
                    
        aggregates = evaluation_aggregation(results)
        averages = evaluation_average(aggregates)
        print('Results:')
        pprint(averages)
        return results, aggregates, averages, discrepancies
                
    def eval_dir(self, dir):
        results = []
        dir = Path(dir)
        for f in tqdm(os.listdir(dir), desc='Evaluating'):
            img = sitk.ReadImage(dir/f)
            pred = sitk.GetImageFromArray(self.pipeline(img, 'MRA' if '_mr_' in f else 'CTA'))
            pred.CopyInformation(img)
            res = evaluation_function(pred, sitk.ReadImage(dir/'location_masks'/f.replace('_0000', '')))
            res['modality'] = 'MRA' if '_mr_' in f else 'CTA'
            results.append(res)
        aggregates_all = evaluation_aggregation(results)
        averages_all = evaluation_average(aggregates_all)
        print('Results:')
        pprint(averages_all)
        return results, aggregates_all, averages_all
    
    def eval_ds(self, testset):
        mapping = {v:k for k, v in location_mapping['labels'].items()}
        results = []
        discrepancies = {}
        for i in tqdm(range(len(testset)), desc='Evaluating'):
            smp = testset[i]
            if not self.use_perf:
                fn = str(testset.src/'images'/f"{smp['id']}_0000.nii.gz")
                try: pred = self.pipeline(fn, smp['modality'], base=self.precomp)
                except: print('skipping', fn);continue
            else: pred = self.pipeline(smp, smp['modality'])
            assert isinstance(pred, np.ndarray)
            if self.wdir is not None:
                os.makedirs(self.wdir/'predictions'/smp['id'].split('.')[0], exist_ok=True)
                gt = sitk.GetImageFromArray(smp['location_mask'])
                p = sitk.GetImageFromArray(pred)
                sitk.WriteImage(gt, self.wdir/'predictions'/smp['id'].split('.')[0]/'gt.nii.gz')
                sitk.WriteImage(p, self.wdir/'predictions'/smp['id'].split('.')[0]/'pred.nii.gz')
            res = evaluation_function(sitk.GetImageFromArray(pred), sitk.GetImageFromArray(smp['location_mask']))
            res['modality'] = smp['modality']
            results.append(res)
            
            if self.use_perf:
                cc, n = label(smp['location_mask'])
                for i in range(1, n+1):
                    slc = cc==i
                    gt_v=np.median(smp['location_mask'][slc])
                    pred_v=np.median(pred[slc])
                    if gt_v!=pred_v:
                        discrepancies[smp['id']+f'_{i}']={'GT':mapping[gt_v],'Predicted':mapping[pred_v]}
            else:
                cc, n = label(smp['location_mask'])
                for i in range(1, n+1):
                    slc = cc==i
                    gt_v=np.median(smp['location_mask'][slc])
                    pred_v=np.median(pred[slc])
                    if gt_v!=pred_v:
                        discrepancies[smp['id']+f'_{i}']={'GT':mapping[gt_v],'Predicted':mapping[pred_v]}
            
        aggregates = evaluation_aggregation(results)
        averages = evaluation_average(aggregates)
        print('Results:')
        pprint(averages)
        return results, aggregates, averages, discrepancies
    
    def eval_ds_demo(self, testset, label_correct):
        mapping = {v:k for k, v in location_mapping['labels'].items()}
        results = []
        discrepancies = {}
        values = []
        for i in tqdm(range(len(testset)), desc='Evaluating'):
            smp = testset[i]
            values += np.unique(smp['location_mask']).tolist()
            pred = ((smp['location_mask']==label_correct)*label_correct).astype(np.uint8)
            assert isinstance(pred, np.ndarray)
            res = evaluation_function(sitk.GetImageFromArray(pred), sitk.GetImageFromArray(smp['location_mask']))
            res['modality'] = smp['modality']
            results.append(res)
            
            if self.use_perf:
                cc, n = label(smp['location_mask'])
                for i in range(1, n+1):
                    slc = cc==i
                    gt_v=np.median(smp['location_mask'][slc])
                    pred_v=np.median(pred[slc])
                    if gt_v!=pred_v:
                        discrepancies[smp['id']+f'_{i}']={'GT':mapping[gt_v],'Predicted':mapping[pred_v]}
            
            
        aggregates = evaluation_aggregation(results)
        averages = evaluation_average(aggregates)
        print('Results:')
        pprint(averages)
        pprint(Counter([lbl for lbl in values if lbl != 0]))
        return results, aggregates, averages, discrepancies
    
    def re_eval_by_modality(self, results, path):
        path = Path(path)
        
        mr = [smp for smp in results if smp['modality']=='MRA']
        ct = [smp for smp in results if smp['modality']=='CTA']
        
        aggregates_mr = evaluation_aggregation(mr)
        averages_mr = evaluation_average(aggregates_mr)
        print('Results MRA:')
        pprint(averages_mr)
        self.plot(aggregates_mr, path/'per_cls_mr', 'Per Class Aggregates MRA')
        with open(path/'per_cls_mr.json', 'w') as f:
            json.dump(averages_mr, f, indent=4)
        
        aggregates_ct = evaluation_aggregation(ct)
        averages_ct = evaluation_average(aggregates_ct)
        print('Results CTA:')
        pprint(averages_ct)
        self.plot(aggregates_ct, path/'per_cls_ct', 'Per Class Aggregates CTA')
        with open(path/'per_cls_ct.json', 'w') as f:
            json.dump(averages_ct, f, indent=4)
    
    def plot(self, aggregates, path, title='Per Class Aggregates'):
        mapping = {v:k for k, v in location_mapping['labels'].items()}
        metrics_by_class = {}
        for i in range(1, 53):
            cur_vals = {
                'Precision': aggregates[f'PRECISION_{i}'],
                'Recall': aggregates[f'RECALL_{i}'],
                'MCC': aggregates[f'MCC_{i}'],
                'Dice': aggregates[f'DICE_{i}'],
                'VolSim': aggregates[f'VOLSIM_{i}'],
                'HD95': aggregates[f'HD95_{i}']
            }
            metrics_by_class[mapping[i]]=cur_vals
        
        heatmap(metrics_by_class, title, str(path)+"_heatmap.png")
        spider(metrics_by_class, title, str(path)+"_cluttered_spider.png")
        decluttered_spider(metrics_by_class, title, str(path)+"_spider.png")