# import os
# from engine.dehaze import train
# from data.uieb import UIEBTrain, UIEBValid
# from torch.utils.data import DataLoader
# from timm.optim import AdamW
# from timm.scheduler import CosineLRScheduler
# from model.base import CLCC
# from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
# from torch.cuda.amp import GradScaler


# def configuration_dataloader(hparams, stage_index):
#     train_dataset = UIEBTrain(
#         folder=hparams['data']['train_path'],
#         size=hparams['data']['train_img_size'][stage_index]
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader


# def configuration_dataloader2(hparams, stage_index):
#     train_dataset = UIEBValid(
#         folder=hparams['data']['train_path'],
#         size=256
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader


# def configuration_optimizer(model, hparams):
#     total_epochs = sum(hparams['train']['stage_epochs'])
#     optimizer = AdamW(
#         params=model.parameters(),
#         lr=hparams['optim']['lr_init'],
#         weight_decay=hparams['optim']['weight_decay']
#     )
#     scheduler = CosineLRScheduler(
#         optimizer=optimizer,
#         t_initial=(sum(hparams['train']['stage_epochs']) // len(hparams['train']['stage_epochs'])
#                    if hparams['optim']['use_cycle_limit'] else sum(hparams['train']['stage_epochs'])),
#         lr_min=hparams['optim']['lr_min'],
#         cycle_limit=len(hparams['train']['stage_epochs']) if hparams['optim']['use_cycle_limit'] else 1,
#         cycle_decay=hparams['optim']['cycle_decay'],
#         warmup_t=hparams['optim']['warmup_epochs'],
#         warmup_lr_init=hparams['optim']['lr_min']
#     )
#     return optimizer, scheduler


# if __name__ == '__main__':
#     args = parse_yaml(r'./config.yaml')
#     base_path = os.path.join(args['train']['save_dir'],
#                              args['train']['model_name'],
#                              args['train']['task_name'])

#     model = CLCC(64, 3, 3).cuda()
#     scaler = GradScaler()
#     logger = Logger(os.path.join(base_path, 'tensorboard'))
#     optimizer, scheduler = configuration_optimizer(model, args)
#     save_dict_as_yaml(args, base_path)
#     print_params_and_macs(model)

#     for i in range(len(args['train']['stage_epochs'])):
#         print('\033[92m\nStart Stage {}'.format(i + 1))
#         if i != 0:
#             args['train']['resume'] = True
#         train_loader, valid_loader = configuration_dataloader2(args, i)
#         train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
#         print('\033[92mEndStage {}\n'.format(i + 1))



# import os
# import torch
# from engine.dehaze import train
# from data.uieb import UIEBTrain, UIEBValid
# from torch.utils.data import DataLoader
# from timm.optim import AdamW
# from timm.scheduler import CosineLRScheduler
# from model.base import CLCC
# from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
# from torch.cuda.amp import GradScaler


# def configuration_dataloader(hparams, stage_index):
#     train_dataset = UIEBTrain(
#         folder=hparams['data']['train_path'],
#         size=hparams['data']['train_img_size'][stage_index]
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader


# def configuration_dataloader2(hparams, stage_index):
#     train_dataset = UIEBValid(
#         folder=hparams['data']['train_path'],
#         size=256
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader


# def configuration_optimizer(model, hparams):
#     total_epochs = sum(hparams['train']['stage_epochs'])
#     optimizer = AdamW(
#         params=model.parameters(),
#         lr=hparams['optim']['lr_init'],
#         weight_decay=hparams['optim']['weight_decay']
#     )
#     scheduler = CosineLRScheduler(
#         optimizer=optimizer,
#         t_initial=(sum(hparams['train']['stage_epochs']) // len(hparams['train']['stage_epochs'])
#                    if hparams['optim']['use_cycle_limit'] else sum(hparams['train']['stage_epochs'])),
#         lr_min=hparams['optim']['lr_min'],
#         cycle_limit=len(hparams['train']['stage_epochs']) if hparams['optim']['use_cycle_limit'] else 1,
#         cycle_decay=hparams['optim']['cycle_decay'],
#         warmup_t=hparams['optim']['warmup_epochs'],
#         warmup_lr_init=hparams['optim']['lr_min']
#     )
#     return optimizer, scheduler


# if __name__ == '__main__':
#     args = parse_yaml(r'./config.yaml')
#     base_path = os.path.join(args['train']['save_dir'],
#                              args['train']['model_name'],
#                              args['train']['task_name'])

#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA GPU not available. Please enable GPU to run this script.")
#     device = torch.device("cuda")
#     print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

#     model = CLCC(64, 3, 3).to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
#         model = torch.nn.DataParallel(model)

#     scaler = GradScaler()
#     logger = Logger(os.path.join(base_path, 'tensorboard'))
#     optimizer, scheduler = configuration_optimizer(model, args)
#     save_dict_as_yaml(args, base_path)
#     print_params_and_macs(model)

#     for i in range(len(args['train']['stage_epochs'])):
#         print('\033[92m\nStart Stage {}'.format(i + 1))
#         if i != 0:
#             args['train']['resume'] = True
#         train_loader, valid_loader = configuration_dataloader2(args, i)
#         train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
#         print('\033[92mEndStage {}\n'.format(i + 1))



# # train_with_metrics.py
# import os
# import time
# import torch
# import yaml
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from engine.dehaze import train  # existing train (unchanged)
# from data.uieb import UIEBTrain, UIEBValid
# from torch.utils.data import DataLoader
# from timm.optim import AdamW
# from timm.scheduler import CosineLRScheduler
# from model.base import CLCC
# from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
# from torch.cuda.amp import GradScaler
# from sklearn.metrics import precision_recall_fscore_support
# from collections import defaultdict
# import csv

# # ---------------------------
# # Utility: simplified IoU / mAP@0.5 for bounding-box-style outputs
# # Note: expects boxes in [x1,y1,x2,y2] absolute pixel coords
# # ---------------------------
# def iou(boxA, boxB):
#     # box: [x1,y1,x2,y2]
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH
#     boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
#     boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
#     union = boxAArea + boxBArea - interArea
#     if union == 0:
#         return 0.0
#     return interArea / union

# def average_precision_at_iou(det_boxes, det_scores, det_labels, gt_boxes, gt_labels, iou_threshold=0.5):
#     # Simple per-class AP calculation following PASCAL VOC-style matching at single IoU threshold
#     # det_* and gt_* are lists for a single image
#     # Returns AP per class (dict)
#     class_ap = {}
#     classes = set(gt_labels) | set(det_labels)
#     for cls in classes:
#         # collect detections and gts of this class across the dataset where this function will be called per dataset
#         # For single-image helpers, we'll leave at image-level; the wrapper will aggregate across images.
#         pass
#     # This helper is not used directly; mAP computation is implemented in evaluate_map below.
#     return {}

# def evaluate_map(all_detections, all_gts, iou_thr=0.5):
#     # all_detections, all_gts: lists (per image) of dict {boxes: [N,4], scores: [N], labels: [N]}
#     # Implementation: for each class compute AP using sorted detections across dataset (simple VOC2007 style)
#     # Returns mAP (mean AP across classes) and per-class AP dict
#     per_class_dets = defaultdict(list)  # cls -> list of (image_idx, score, box)
#     per_class_gts = defaultdict(lambda: defaultdict(list))  # cls -> image_idx -> list of boxes (matched flags later)
#     n_images = len(all_gts)
#     for img_idx, det in enumerate(all_detections):
#         boxes = det.get('boxes', [])
#         scores = det.get('scores', [])
#         labels = det.get('labels', [])
#         for b, s, l in zip(boxes, scores, labels):
#             per_class_dets[int(l)].append((img_idx, float(s), np.array(b, dtype=float)))
#     for img_idx, gt in enumerate(all_gts):
#         boxes = gt.get('boxes', [])
#         labels = gt.get('labels', [])
#         for b, l in zip(boxes, labels):
#             per_class_gts[int(l)][img_idx].append(np.array(b, dtype=float))

#     ap_per_class = {}
#     for cls, det_list in per_class_dets.items():
#         # sort by score desc
#         det_list_sorted = sorted(det_list, key=lambda x: x[1], reverse=True)
#         npos = 0
#         for img_idx, boxes in per_class_gts[cls].items():
#             npos += len(boxes)
#         if npos == 0:
#             ap_per_class[cls] = 0.0
#             continue
#         tp = np.zeros(len(det_list_sorted))
#         fp = np.zeros(len(det_list_sorted))
#         # matched flags per gt box
#         matched = {img_idx: np.zeros(len(per_class_gts[cls][img_idx]), dtype=bool) for img_idx in per_class_gts[cls]}
#         for d_i, (img_idx, score, dbox) in enumerate(det_list_sorted):
#             gboxes = per_class_gts[cls].get(img_idx, [])
#             best_iou = 0.0
#             best_gt_idx = -1
#             for gt_i, gbox in enumerate(gboxes):
#                 cur_iou = iou(dbox, gbox)
#                 if cur_iou > best_iou:
#                     best_iou = cur_iou
#                     best_gt_idx = gt_i
#             if best_iou >= iou_thr:
#                 if not matched.get(img_idx, np.array([]))[best_gt_idx]:
#                     tp[d_i] = 1
#                     matched[img_idx][best_gt_idx] = True
#                 else:
#                     fp[d_i] = 1
#             else:
#                 fp[d_i] = 1
#         # compute precision-recall curve
#         fp_cum = np.cumsum(fp)
#         tp_cum = np.cumsum(tp)
#         rec = tp_cum / float(npos)
#         prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
#         # compute AP as area under PR via interpolation (VOC2007 11-point can be used; we'll use trapezoid)
#         # but ensure monotonic precision
#         mpre = np.concatenate(([0.0], prec, [0.0]))
#         mrec = np.concatenate(([0.0], rec, [1.0]))
#         for i in range(len(mpre) - 1, 0, -1):
#             if mpre[i - 1] < mpre[i]:
#                 mpre[i - 1] = mpre[i]
#         # integrate
#         idx = np.where(mrec[1:] != mrec[:-1])[0]
#         ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
#         ap_per_class[cls] = float(ap)
#     # include classes present only in gts but with no detections
#     for cls in per_class_gts.keys():
#         if cls not in ap_per_class:
#             ap_per_class[cls] = 0.0
#     if len(ap_per_class) == 0:
#         return 0.0, {}
#     mAP = float(np.mean(list(ap_per_class.values())))
#     return mAP * 100.0, {k: v * 100.0 for k, v in ap_per_class.items()}

# # ---------------------------
# # DataLoader configuration (kept same)
# # ---------------------------
# def configuration_dataloader(hparams, stage_index):
#     train_dataset = UIEBTrain(
#         folder=hparams['data']['train_path'],
#         size=hparams['data']['train_img_size'][stage_index]
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader

# def configuration_dataloader2(hparams, stage_index):
#     train_dataset = UIEBValid(
#         folder=hparams['data']['train_path'],
#         size=256
#     )
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=hparams['data']['train_batch_size'][stage_index],
#         shuffle=True,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     valid_dataset = UIEBValid(
#         folder=hparams['data']['valid_path'],
#         size=256
#     )
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=1,
#         num_workers=hparams['data']['num_workers'],
#         pin_memory=hparams['data']['pin_memory']
#     )
#     return train_loader, valid_loader

# def configuration_optimizer(model, hparams):
#     total_epochs = sum(hparams['train']['stage_epochs'])
#     optimizer = AdamW(
#         params=model.parameters(),
#         lr=hparams['optim']['lr_init'],
#         weight_decay=hparams['optim']['weight_decay']
#     )
#     scheduler = CosineLRScheduler(
#         optimizer=optimizer,
#         t_initial=(sum(hparams['train']['stage_epochs']) // len(hparams['train']['stage_epochs'])
#                    if hparams['optim']['use_cycle_limit'] else sum(hparams['train']['stage_epochs'])),
#         lr_min=hparams['optim']['lr_min'],
#         cycle_limit=len(hparams['train']['stage_epochs']) if hparams['optim']['use_cycle_limit'] else 1,
#         cycle_decay=hparams['optim']['cycle_decay'],
#         warmup_t=hparams['optim']['warmup_epochs'],
#         warmup_lr_init=hparams['optim']['lr_min']
#     )
#     return optimizer, scheduler

# # ---------------------------
# # Evaluation function (runs on validation loader)
# # ---------------------------
# @torch.no_grad()
# def evaluate_model(model, device, valid_loader, expected_det_format=True):
#     model.eval()
#     timings = {'preprocess': [], 'inference': [], 'postprocess': []}
#     all_gts = []
#     all_dets = []
#     y_true_all = []
#     y_pred_all = []
#     losses = defaultdict(list)  # if model returns losses in a validation forward, capture them

#     for batch_idx, sample in enumerate(valid_loader):
#         # sample can be dict or tuple - be flexible
#         t0 = time.time()
#         # Preprocess: move tensors to device if they exist
#         preprocess_start = time.time()
#         inputs = None
#         gts = None
#         # try common conventions
#         if isinstance(sample, dict):
#             # typical names
#             inputs = sample.get('image') or sample.get('input') or sample.get('img') or sample.get('images')
#             gts = sample.get('target') or sample.get('label') or sample.get('labels') or sample.get('gt') or sample.get('gts')
#         elif isinstance(sample, (tuple, list)):
#             if len(sample) >= 1:
#                 inputs = sample[0]
#             if len(sample) >= 2:
#                 gts = sample[1]
#         # Move inputs to device if tensor
#         if torch.is_tensor(inputs):
#             inputs = inputs.to(device, non_blocking=True)
#         preprocess_end = time.time()
#         timings['preprocess'].append(preprocess_end - preprocess_start)

#         # inference
#         inf_start = time.time()
#         # We expect model to return either:
#         # 1) For detectors: dict with keys 'boxes','scores','labels' or list-of-dicts per image
#         # 2) For image-to-image tasks: returned tensor
#         out = model(inputs) if inputs is not None else model()
#         inf_end = time.time()
#         timings['inference'].append(inf_end - inf_start)

#         # postprocess (format model output to detection-format if possible)
#         post_start = time.time()
#         # Attempt to extract detection-like outputs
#         det_entry = {'boxes': [], 'scores': [], 'labels': []}
#         gt_entry = {'boxes': [], 'labels': []}

#         # If model returned dict or list/dict with keys
#         if isinstance(out, dict):
#             # common keys
#             if 'boxes' in out or 'scores' in out or 'labels' in out:
#                 # assume single-image batch
#                 boxes = out.get('boxes', [])
#                 scores = out.get('scores', [])
#                 labels = out.get('labels', [])
#                 # convert tensors to cpu numpy
#                 if torch.is_tensor(boxes):
#                     boxes = boxes.detach().cpu().numpy().tolist()
#                 if torch.is_tensor(scores):
#                     scores = scores.detach().cpu().numpy().tolist()
#                 if torch.is_tensor(labels):
#                     labels = labels.detach().cpu().numpy().astype(int).tolist()
#                 det_entry['boxes'] = boxes
#                 det_entry['scores'] = scores
#                 det_entry['labels'] = labels
#             else:
#                 # maybe single image tensor output; skip detection metrics
#                 det_entry = None
#         elif isinstance(out, (list, tuple)):
#             # could be list of detection dicts per image
#             first = out[0] if len(out) > 0 else None
#             if isinstance(first, dict) and ('boxes' in first or 'scores' in first):
#                 # assume list of dicts
#                 first = first
#                 boxes = first.get('boxes', [])
#                 scores = first.get('scores', [])
#                 labels = first.get('labels', [])
#                 if torch.is_tensor(boxes):
#                     boxes = boxes.detach().cpu().numpy().tolist()
#                 if torch.is_tensor(scores):
#                     scores = scores.detach().cpu().numpy().tolist()
#                 if torch.is_tensor(labels):
#                     labels = labels.detach().cpu().numpy().astype(int).tolist()
#                 det_entry['boxes'] = boxes
#                 det_entry['scores'] = scores
#                 det_entry['labels'] = labels
#             else:
#                 det_entry = None
#         elif torch.is_tensor(out):
#             # image tensor; not detection
#             det_entry = None
#         else:
#             det_entry = None

#         # ground-truth extraction
#         if gts is not None:
#             if isinstance(gts, dict):
#                 gboxes = gts.get('boxes', [])
#                 glabels = gts.get('labels', [])
#                 if torch.is_tensor(gboxes):
#                     gboxes = gboxes.detach().cpu().numpy().tolist()
#                 if torch.is_tensor(glabels):
#                     glabels = glabels.detach().cpu().numpy().astype(int).tolist()
#                 gt_entry['boxes'] = gboxes
#                 gt_entry['labels'] = glabels
#             elif isinstance(gts, (list, tuple)):
#                 # assume (boxes, labels)
#                 if len(gts) >= 2:
#                     gboxes, glabels = gts[0], gts[1]
#                     if torch.is_tensor(gboxes):
#                         gboxes = gboxes.detach().cpu().numpy().tolist()
#                     if torch.is_tensor(glabels):
#                         glabels = glabels.detach().cpu().numpy().astype(int).tolist()
#                     gt_entry['boxes'] = gboxes
#                     gt_entry['labels'] = glabels
#                 else:
#                     gt_entry = {}
#             elif torch.is_tensor(gts):
#                 # probably segmentation or image target - skip detection metrics
#                 gt_entry = {}
#         else:
#             gt_entry = {}

#         post_end = time.time()
#         timings['postprocess'].append(post_end - post_start)

#         # collect
#         if det_entry is not None:
#             all_dets.append(det_entry)
#         else:
#             all_dets.append({'boxes': [], 'scores': [], 'labels': []})
#         all_gts.append(gt_entry)

#         # For classification-like ground truth/outputs (fallback), try to capture labels if present
#         # If gts is a tensor of labels and model output is logits, we can compute precision/recall
#         if torch.is_tensor(gts) and torch.is_tensor(out):
#             # flatten
#             y_true = gts.detach().cpu().numpy().ravel().tolist()
#             if out.dim() > 1:
#                 y_pred = out.detach().cpu().argmax(dim=1).ravel().cpu().numpy().tolist()
#             else:
#                 y_pred = (out.detach().cpu().numpy() > 0.5).astype(int).ravel().tolist()
#             y_true_all.extend(y_true)
#             y_pred_all.extend(y_pred)

#     # aggregate timings
#     timings_agg = {k: (np.mean(v) * 1000.0 if len(v) else 0.0) for k, v in timings.items()}  # ms
#     total_per_image_ms = sum(timings_agg.values())
#     fps = 1000.0 / total_per_image_ms if total_per_image_ms > 0 else 0.0

#     # compute classification-type precision/recall if labels captured
#     prec, rec, f1 = None, None, None
#     if len(y_true_all) and len(y_pred_all):
#         p, r, f, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='binary', zero_division=0)
#         prec = float(p * 100.0)
#         rec = float(r * 100.0)
#         f1 = float(f)
#     # compute detection mAP@0.5 if possible
#     mAP = None
#     per_class_ap = {}
#     # check if gts contain boxes/labels info
#     any_gt_boxes = any(len(g.get('boxes', [])) > 0 for g in all_gts)
#     any_det_boxes = any(len(d.get('boxes', [])) > 0 for d in all_dets)
#     if any_gt_boxes and any_det_boxes:
#         mAP, per_class_ap = evaluate_map(all_dets, all_gts, iou_thr=0.5)
#         # we can also compute aggregate precision/recall across detections: treat detections with score>threshold as positive
#         # But for now, provide mAP and per-class AP
#     results = {
#         'timings_ms': timings_agg,
#         'total_per_image_ms': total_per_image_ms,
#         'fps': fps,
#         'precision_cls_pct': prec,
#         'recall_cls_pct': rec,
#         'f1_cls': f1,
#         'mAP50_pct': mAP,
#         'per_class_ap_pct': per_class_ap
#     }
#     return results

# # ---------------------------
# # Main: keeps training loop intact, adds evaluation, graphs, PDF output
# # ---------------------------
# if __name__ == '__main__':
#     args = parse_yaml(r'./config.yaml')
#     base_path = os.path.join(args['train']['save_dir'],
#                              args['train']['model_name'],
#                              args['train']['task_name'])

#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA GPU not available. Please enable GPU to run this script.")
#     device = torch.device("cuda")
#     print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

#     model = CLCC(64, 3, 3).to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
#         model = torch.nn.DataParallel(model)

#     scaler = GradScaler()
#     logger = Logger(os.path.join(base_path, 'tensorboard'))
#     optimizer, scheduler = configuration_optimizer(model, args)
#     save_dict_as_yaml(args, base_path)
#     print_params_and_macs(model)

#     # trackers for stage metrics/losses
#     stages_metrics = []
#     stages_losses = []  # optional: if train returns per-stage loss info
#     csv_log_path = os.path.join(base_path, 'training_metrics.csv')
#     os.makedirs(base_path, exist_ok=True)

#     # create CSV header
#     with open(csv_log_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([
#             'stage', 'precision_cls_pct', 'recall_cls_pct', 'f1_cls', 'mAP50_pct',
#             'timing_preprocess_ms', 'timing_inference_ms', 'timing_postprocess_ms',
#             'total_per_image_ms', 'fps'
#         ])

#     for i in range(len(args['train']['stage_epochs'])):
#         print('\033[92m\nStart Stage {}'.format(i + 1))
#         if i != 0:
#             args['train']['resume'] = True
#         train_loader, valid_loader = configuration_dataloader2(args, i)

#         # Call original train - keep same signature / behavior. Try to capture returned stats if any.
#         train_return = train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
#         # If train returns training losses/stats, capture them for plotting later
#         stage_loss_info = None
#         if isinstance(train_return, dict):
#             # assume keys like 'train_losses', 'val_losses'
#             stage_loss_info = train_return.get('losses') or train_return.get('train_losses') or train_return
#             print("Captured returned training stats from train() for this stage.")
#             stages_losses.append({'stage': i + 1, 'loss_info': stage_loss_info})
#         else:
#             stages_losses.append({'stage': i + 1, 'loss_info': None})

#         # Now evaluate model on validation loader (measures timings, computes detection/class metrics where possible)
#         print(f"Evaluating stage {i + 1} on validation set for metrics/timings...")
#         eval_results = evaluate_model(model, device, valid_loader)

#         # Save eval results to disk (yaml)
#         eval_yaml_path = os.path.join(base_path, f'stage_{i+1}_eval.yaml')
#         with open(eval_yaml_path, 'w') as f:
#             yaml.safe_dump(eval_results, f, sort_keys=False)

#         stages_metrics.append({'stage': i + 1, **eval_results})

#         # Append to CSV
#         with open(csv_log_path, 'a', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([
#                 i + 1,
#                 eval_results.get('precision_cls_pct'),
#                 eval_results.get('recall_cls_pct'),
#                 eval_results.get('f1_cls'),
#                 eval_results.get('mAP50_pct'),
#                 eval_results['timings_ms'].get('preprocess'),
#                 eval_results['timings_ms'].get('inference'),
#                 eval_results['timings_ms'].get('postprocess'),
#                 eval_results.get('total_per_image_ms'),
#                 eval_results.get('fps')
#             ])

#         print('\033[92mEndStage {}\n'.format(i + 1))

#     # After all stages: plot comparisons and produce PDF
#     print("Compiling graphs and PDF report...")

#     stages = [m['stage'] for m in stages_metrics]
#     precision = [m.get('precision_cls_pct') or 0.0 for m in stages_metrics]
#     recall = [m.get('recall_cls_pct') or 0.0 for m in stages_metrics]
#     f1s = [m.get('f1_cls') or 0.0 for m in stages_metrics]
#     mAPs = [m.get('mAP50_pct') or 0.0 for m in stages_metrics]
#     preprocess_ms = [m['timings_ms'].get('preprocess', 0.0) for m in stages_metrics]
#     inference_ms = [m['timings_ms'].get('inference', 0.0) for m in stages_metrics]
#     postprocess_ms = [m['timings_ms'].get('postprocess', 0.0) for m in stages_metrics]
#     fps_list = [m.get('fps', 0.0) for m in stages_metrics]

#     pdf_path = os.path.join(base_path, 'training_report.pdf')
#     with PdfPages(pdf_path) as pdf:
#         # Plot: Precision/Recall/F1
#         plt.figure(figsize=(8, 5))
#         plt.plot(stages, precision, marker='o', label='Precision (%)')
#         plt.plot(stages, recall, marker='o', label='Recall (%)')
#         plt.plot(stages, f1s, marker='o', label='F1')
#         plt.title('Precision / Recall / F1 per Stage')
#         plt.xlabel('Stage')
#         plt.xticks(stages)
#         plt.grid(True)
#         plt.legend()
#         pdf.savefig()
#         plt.close()

#         # Plot: mAP@0.5
#         plt.figure(figsize=(8, 5))
#         plt.plot(stages, mAPs, marker='o', label='mAP@0.5 (%)')
#         plt.title('mAP@0.5 per Stage')
#         plt.xlabel('Stage')
#         plt.xticks(stages)
#         plt.grid(True)
#         plt.legend()
#         pdf.savefig()
#         plt.close()

#         # Plot: timings stacked or separate
#         plt.figure(figsize=(8, 5))
#         plt.plot(stages, preprocess_ms, marker='o', label='Preprocess (ms)')
#         plt.plot(stages, inference_ms, marker='o', label='Inference (ms)')
#         plt.plot(stages, postprocess_ms, marker='o', label='Postprocess (ms)')
#         plt.title('Per-image timings per Stage (ms)')
#         plt.xlabel('Stage')
#         plt.xticks(stages)
#         plt.grid(True)
#         plt.legend()
#         pdf.savefig()
#         plt.close()

#         # FPS plot
#         plt.figure(figsize=(8, 5))
#         plt.plot(stages, fps_list, marker='o', label='FPS')
#         plt.axhline(30, linestyle='--', label='Real-time threshold (30 FPS)')
#         plt.title('FPS per Stage')
#         plt.xlabel('Stage')
#         plt.xticks(stages)
#         plt.grid(True)
#         plt.legend()
#         pdf.savefig()
#         plt.close()

#         # If any stage returned loss info, attempt to plot losses
#         any_losses = any(s.get('loss_info') for s in stages_losses)
#         if any_losses:
#             # Try to plot training loss decrease if structured as list/array per epoch
#             for s in stages_losses:
#                 if s.get('loss_info') and isinstance(s['loss_info'], dict):
#                     # try keys
#                     keys = list(s['loss_info'].keys())
#                     for k in keys:
#                         try:
#                             vals = s['loss_info'][k]
#                             if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) > 1:
#                                 plt.figure(figsize=(8, 5))
#                                 plt.plot(range(1, len(vals) + 1), vals, marker='o')
#                                 plt.title(f"Stage {s['stage']} - {k}")
#                                 plt.xlabel('Epoch (within stage)')
#                                 plt.ylabel(k)
#                                 plt.grid(True)
#                                 pdf.savefig()
#                                 plt.close()
#                         except Exception:
#                             continue

#     print(f"PDF report saved to: {pdf_path}")
#     print(f"CSV metrics log saved to: {csv_log_path}")
#     print("Done.")


import os
import torch
from engine.dehaze import train
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from model.base import CLCC
from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
from torch.cuda.amp import GradScaler


def configuration_dataloader(hparams, stage_index):
    train_dataset = UIEBTrain(
        folder=hparams['data']['train_path'],
        size=hparams['data']['train_img_size'][stage_index]
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_dataloader2(hparams, stage_index):
    train_dataset = UIEBValid(
        folder=hparams['data']['train_path'],
        size=256
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_optimizer(model, hparams):
    total_epochs = sum(hparams['train']['stage_epochs'])
    optimizer = AdamW(
        params=model.parameters(),
        lr=hparams['optim']['lr_init'],
        weight_decay=hparams['optim']['weight_decay']
    )
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(sum(hparams['train']['stage_epochs']) // len(hparams['train']['stage_epochs'])
                   if hparams['optim']['use_cycle_limit'] else sum(hparams['train']['stage_epochs'])),
        lr_min=hparams['optim']['lr_min'],
        cycle_limit=len(hparams['train']['stage_epochs']) if hparams['optim']['use_cycle_limit'] else 1,
        cycle_decay=hparams['optim']['cycle_decay'],
        warmup_t=hparams['optim']['warmup_epochs'],
        warmup_lr_init=hparams['optim']['lr_min']
    )
    return optimizer, scheduler


if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    
    args = parse_yaml(r'./config.yaml')
    base_path = os.path.join(args['train']['save_dir'],
                             args['train']['model_name'],
                             args['train']['task_name'])

    model = CLCC(64, 3, 3).to(device)
    scaler = GradScaler()
    logger = Logger(os.path.join(base_path, 'tensorboard'))
    optimizer, scheduler = configuration_optimizer(model, args)
    save_dict_as_yaml(args, base_path)
    print_params_and_macs(model)

    for i in range(len(args['train']['stage_epochs'])):
        print('\033[92m\nStart Stage {}'.format(i + 1))
        if i != 0:
            args['train']['resume'] = True
        train_loader, valid_loader = configuration_dataloader2(args, i)
        train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
        print('\033[92mEndStage {}\n'.format(i + 1))