import argparse
import motmetrics as mm
import numpy as np
import torch
import h5py

def mot(
        pred,
        pred_conf,
        pred_ids,
        gt,
        gt_in_camera,
        gt_ids,
        distance_metric='iou',
        max_distance=0.5,
        pred_conf_threshold=0.5
):
    """
        Args (all numpy arrays):
            pred: (B, T, N, D) prediction boxes or centers depending on distance_metric
            pred_conf: (B, T, N, 1) confidence of box (z_pres)
            pred_ids: (B, T, N)
            gt: (B, T, N, D) ground truth boxes or centers depending on distance_metric
            gt_in_camera: (B, T, N) whether or not the gt box is in the camera
            gt_ids: (B, T, N)
            distance_metric: {'iou', 'euclidean'}
                if 'iou', D = 4 => [center_x, center_y, width, height] normalized between 0 and 1
                if 'euclidean', D = 2 => [center_x, center_y] normalized between 0 and 1
            max_distance: maximum distance for a match
            pred_conf_threshold: z_pres threshold for a prediction to be counted
    """
    assert distance_metric in ['iou', 'euclidean']
    B, T, N, _ = pred.shape

    accumulators = []
    for b in range(B):
        acc = mm.MOTAccumulator(auto_id=True)
        for t in range(T):
            frame_pred = []
            frame_pred_ids = []
            frame_gt = []
            frame_gt_ids = []
            for n in range(pred_conf.shape[2]):
                if pred_conf[b, t, n, 0] > pred_conf_threshold:
                    frame_pred.append(pred[b, t, n])
                    frame_pred_ids.append(pred_ids[b, t, n])

            for n in range(gt_in_camera.shape[2]):
                if gt_in_camera[b, t, n]:
                    frame_gt.append(gt[b, t, n])
                    frame_gt_ids.append(gt_ids[b, t, n])

            if distance_metric == 'iou':
                distances = mm.distances.iou_matrix(frame_gt, frame_pred, max_iou=max_distance)
            elif distance_metric == 'euclidean':
                distances = mm.distances.norm2squared_matrix(frame_gt, frame_pred, max_d2=max_distance)
            acc.update(frame_gt_ids, frame_pred_ids, distances)
        accumulators.append(acc)

    #print('acc', len(accumulators))
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        metrics=['mota', 'motp', 'num_switches', 'num_false_positives', 'num_misses', 'num_detections', 'num_objects',],
        # metrics=['mota', 'idf1'],
        names=[str(i) for i in range(len(accumulators))],
        generate_overall=True
    )
    return dict(summary.loc['OVERALL'])

def mean_euclidean_distance(
        pred,
        pred_conf,
        pred_ids,
        gt,
        gt_in_camera,
        gt_ids,
        distance_metric='iou',
        max_distance=0.5,
        pred_conf_threshold=0.5
):
    assert distance_metric in ['iou', 'euclidean']
    B, T, N, _ = pred.shape

    meds_per_batch = []
    for b in range(B):
        acc = mm.MOTAccumulator(auto_id=True)
        meds_per_timestep = []
        gt_pred_pairings = None
        for t in range(T):
            frame_pred = []
            frame_pred_ids = []
            pred_map = {}
            frame_gt = []
            frame_gt_ids = []
            gt_map = {}
            for n in range(pred_conf.shape[2]):
                if pred_conf[b, t, n, 0] > pred_conf_threshold:
                    p = pred[b, t, n]
                    pid = pred_ids[b, t, n]
                    frame_pred.append(p)
                    frame_pred_ids.append(pid)
                    pred_map[pid] = p

            for n in range(gt_in_camera.shape[2]):
                if gt_in_camera[b, t, n]:
                    g = gt[b, t, n]
                    gid = gt_ids[b, t, n]
                    frame_gt.append(g)
                    frame_gt_ids.append(gid)
                    gt_map[gid] = g

            if distance_metric == 'iou':
                distances = mm.distances.iou_matrix(frame_gt, frame_pred, max_iou=max_distance)
            elif distance_metric == 'euclidean':
                distances = mm.distances.norm2squared_matrix(frame_gt, frame_pred, max_d2=max_distance)

            if gt_pred_pairings is None:
                gt_pred_pairings = [(frame_gt_ids[g], frame_pred_ids[p]) for g, p in zip(*mm.lap.linear_sum_assignment(distances))]

            med = 0
            for gt_id, pred_id in gt_pred_pairings:
                if pred_id not in pred_map:
                    #print(f'pred_id {pred_id} disappeared. be sure to set z_pres to 1 after first timestep')
                    med = np.nan
                    break
                curr_med = np.sqrt(((gt_map[gt_id] - pred_map[pred_id])**2).sum())
                med += curr_med
            if len(gt_pred_pairings) > 0:
                meds_per_timestep.append(med / len(gt_pred_pairings))
            else:
                meds_per_timestep.append(np.nan)
        meds_per_batch.append(meds_per_timestep)

    # B, T
    meds_per_batch = np.asarray(meds_per_batch)
    meds_over_time = meds_per_batch.mean(axis=0)
    meds_overall = meds_per_batch.mean()

    return {
        'meds_over_time': meds_over_time,
        'meds_overall': meds_overall,
    }



def msprite_gt_to_boxes(gt_positions, gt_sizes, orig_img_size=64):
    """
    Args:
        gt_positions (all numpy arrays):
            (B, T, max_obj, 2) in pixels: [0, orig_img_size)
            max_obj is the maximum number of objects (currently 10). slots with no objects will be 0
        gt_sizes:
            (B, T, max_obj) half of the actual size of the shape in pixels (actual size is gt_sizes*2)
            max_obj is the maximum number of objects (currently 10). slots with no objects will be 0
    Returns:
        gt_boxes: (B, T, max_obj, 4) where 4 => [center_x, center_y, width, height] normalized between 0 and 1
    """
    B, T, max_obj, pos_d = gt_positions.shape

    gt_positions = gt_positions / orig_img_size

    gt_sizes = (gt_sizes * 2)
    gt_sizes = gt_sizes / orig_img_size
    # (B, T, max_obj, 1)
    gt_sizes = np.expand_dims(gt_sizes, axis=-1)

    return np.concatenate((gt_positions, gt_sizes, gt_sizes), axis=3)

def scalor_pred_to_boxes(z_where):
    """
    Args:
        z_where: (B, T, N, 4) where 4 => [width, height, center_x, center_y]
        z_pres: (B, T, N, 1)
        pres_threshold: only include objects over this threshould
    Returns:
        boxes: (B, T, N, 4) where 4 => [center_x, center_y, width, height] normalized between 0 and 1
    """
    pred_boxes = z_where[:, :, :, [2, 3, 0, 1]].clone()
    pred_boxes[:, :, :, 0:2] = (pred_boxes[:, :, :, 0:2] + 1) / 2
    return pred_boxes

def metrics_from_file(eval_file, metrics):
    with h5py.File(eval_file, 'r') as f:
        pred = f['pred'][:]
        pred_conf = f['pred_conf'][:]
        pred_ids = f['pred_ids'][:]
        gt_positions = f['gt_positions'][:]
        gt_sizes = f['gt_sizes'][:]
        gt_ids = f['gt_ids'][:]
        gt_in_camera = f['gt_in_camera'][:]

        gt = msprite_gt_to_boxes(gt_positions, gt_sizes)

        if 'mot_iou' in metrics:
            print('Computing MOT (IOU)...')
            iou_args = [pred, pred_conf, pred_ids, gt, gt_in_camera, gt_ids]
            iou_summary = mot(*iou_args, distance_metric='iou', max_distance=0.5)
        else:
            iou_summary = None
        # print(iou_summary)

        if 'mot_dist' in metrics:
            print('Computing MOT (distance)...')
            euclidean_args = [pred[:, :, :, :2], pred_conf, pred_ids, gt[:, :, :, :2], gt_in_camera, gt_ids]
            euclidean_summary = mot(*euclidean_args, distance_metric='euclidean', max_distance=1.0)
        else:
            euclidean_summary = None
        # print(euclidean_summary)

        ## med assumes objects do not disappear -- need to fix z_pres after initial frame
        if 'med' in metrics:
            print('Computing MED...')
            med_args = [pred[:, :, :, :2], pred_conf, pred_ids, gt[:, :, :, :2], gt_in_camera, gt_ids]
            med_summary = mean_euclidean_distance(*med_args, distance_metric='euclidean', max_distance=1.0)
        else:
            med_summary = None
        # print(med_summary)

        return iou_summary, euclidean_summary, med_summary



