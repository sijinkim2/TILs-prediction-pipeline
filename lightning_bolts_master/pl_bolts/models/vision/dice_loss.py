"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

from torchvision.transforms import Lambda
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import functional as F


def Class_wise_Dice_score(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01
    pred = F.softmax(input)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels)
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels)
    labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels)

    labels_0 = labels_0.contiguous().view(-1)
    labels_1 = labels_1.contiguous().view(-1)
    labels_2 = labels_2.contiguous().view(-1)

    pred_0 = Lambda(lambda x: x[:, 0, :, :])(pred)
    pred_1 = Lambda(lambda x: x[:, 1, :, :])(pred)
    pred_2 = Lambda(lambda x: x[:, 2, :, :])(pred)

    pred_0 = pred_0.contiguous().view(-1)
    pred_1 = pred_1.contiguous().view(-1)
    pred_2 = pred_2.contiguous().view(-1)

    intersection_0 = (pred_0 * labels_0).sum()
    intersection_1 = (pred_1 * labels_1).sum()
    intersection_2 = (pred_2 * labels_2).sum()
    intersection_SUM = intersection_0 + intersection_1 + intersection_2

    union_0 = (pred_0.sum() + labels_0.sum()) + eps
    union_1 = (pred_1.sum() + labels_1.sum()) + eps
    union_2 = (pred_2.sum() + labels_2.sum()) + eps
    union_SUM = union_0 + union_1 + union_2

    dice_total = (2 * intersection_SUM / union_SUM)
    dice_total_loss = 1 - dice_total

    dice0 = (2 * intersection_0 / union_0)
    dice1 = (2 * intersection_1 / union_1)
    dice2 = (2 * intersection_2 / union_2)
    diceT = dice0 + dice1 + dice2

    bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')
    return dice0, dice1, dice2, dice_total, bce_loss, dice_total_loss, (bce_loss * b) + dice_total_loss

def extract_predictions(probabilities, confidence_threshold):
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]
    return boxes_pred


def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)
    return wanted

def count_tp_fp_fn(pred, gt, prob_threshold, hit_distance):
    # This function counts the number of true-positives, false negatives and false positives in a detection taks.

    # inputs: A numpy array of shape [N,3] for pred and gt. N is the number of detections. The first column is the x-position of the detections.
    # the second column is the y-position of the detections. the third column is the probability associated with detections.
    # The hit-distance is the maximum distance of a detection from the ground-truth to be counted as a true-positive, otherwsie, it will be counted as a false positive.
    # The prob_threshld is the threshold value in which detections with probabilities smaller than the threshold value will be discarded.

    # output: number of ture-positives, false positives, and false negatives.
    if pred.shape[0] > 0:
        pred = pred[np.argwhere(pred[:, 2] >= prob_threshold)[:, 0], :2]
        gt = gt[np.argwhere(gt[:, 2] >= prob_threshold)[:, 0], :2]
    if pred.shape[0] > 0:
        if gt.shape[0] > 0:
            hit = cdist(pred[:, :2], gt)
            hit[hit <= hit_distance] = 1
            hit[hit != 1] = 0
            sum_1 = np.sum(hit, 0)
            sum_2 = np.sum(hit, 1)
            fp = sum(sum_2 == 0)
            fn = sum(sum_1 == 0)
            tp = sum(sum_1 != 0)
        else:
            tp = 0
            fn = 0
            fp = pred.shape[0]
    else:
        tp = 0
        fp = 0
        fn = gt.shape[0]
    return tp, fp, fn


######## Calculating TP,FP,FN for each Patch, Add them all and then calculate Sen,FP-Per-Patch

def FROC(pred, gt, confidence_thresholds, hit_distance):
    # this function calculates the average sensitivity and average number of fp_per_image.

    # inputs: pred and gt are lists containing detections from the ground truth and the predictions from the model. for example:
    # pred = [pred_1, pred_2, pred_3], gt = [gt_1, gt_2, gt_3]
    # pred_1 = np.array((N,3))  pred_2 = np.array((M,3))    pred_3 = np.array((P,3))
    # gt_1 = np.array((N,3))    gt_2 = np.array((M,3))      gt_3 = np.array((P,3))
    # confidence_thresholds is the list of the probability thresholds in calculating the FROC curve. for example:
    # confidence_thresholds = np.linspace(0,1,40)
    # hit-distance: is the maximum distance from the ground truth by which a detection will be counted as a true positive.

    # outputs: returns a list for sensitivies and fps_per_image for each of the threshold values.
    sens = []

    fp_img = []
    for threshold in confidence_thresholds:
        tps = []
        fps = []
        fns = []
        for N in range(len(gt)):
            tp, fp, fn = count_tp_fp_fn(pred[N], gt[N], threshold, hit_distance)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        tps = sum(tps)
        fp_per_img = np.mean(fps)
        fps = sum(fps)
        fns = sum(fns)
        if tps != 0:
            sens.append(tps / (tps + fns))
        else:
            sens.append(0)
        fp_img.append(fp_per_img)
    return sens, fp_img


def compute_froc_score(fps_per_image, total_sensitivity):

    interp_sens = np.interp((10,20,50,100,200,300), fps_per_image[::-1], total_sensitivity[::-1])
    return np.mean(interp_sens)

def Class_Wise_TIL_Detection_FROC(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01

    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 12
    confidence_threshold = 0.1
    predicted_detections = []
    ground_truth_detections = []

    preds = F.softmax(input)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels) #background class of targets
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels) #TIL class of targets
    labels_00 = labels_0.detach().cpu().numpy()
    labels_11 = labels_1.detach().cpu().numpy()


    preds_0 = Lambda(lambda x: x[:, 0, :, :])(preds) #background class of predictions
    preds_1 = Lambda(lambda x: x[:, 1, :, :])(preds) #TIL class of predictions
    preds_00 = preds_0.detach().cpu().numpy()
    preds_11 = preds_1.detach().cpu().numpy()

    # get intersection
    intersection_0 = (preds_0 * labels_0).sum()
    intersection_1 = (preds_1 * labels_1).sum()
    intersection_SUM = intersection_0 + intersection_1

    # get union
    union_0 = (preds_0.sum() + labels_0.sum()) + eps
    union_1 = (preds_1.sum() + labels_1.sum()) + eps
    union_SUM = union_0 + union_1

    # get dice total score and dice loss
    dice_total = (2 * intersection_SUM / union_SUM)
    dice_total_loss = 1 - dice_total

    #dice0 = (2 * intersection_0 / union_0)
    #dice1 = (2 * intersection_1 / union_1)

    bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')


    for i in range(len(preds_11)):
        #print(i)
        temp1 = extract_predictions(preds_11[i], confidence_threshold=confidence_threshold)
        predicted_detections.append(non_max_supression_distance(temp1, distance_threshold=12))
    del temp1

    for i in range(len(labels_11)):
        #print(i)
        temp2 = extract_predictions(labels_11[i], confidence_threshold=0.1)
        ground_truth_detections.append(non_max_supression_distance(temp2, distance_threshold=distance_threshold))
    del temp2

    print(ground_truth_detections[0].shape)
    sensitivity, fps_image = FROC(predicted_detections, ground_truth_detections, confidence_thresholds, 8)
    froc_score = compute_froc_score(fps_image, sensitivity)
    froc_loss = 1 - froc_score

    #print("                                                             %s" % sensitivity[0])
    #fps_image = int(sum(fps_image)/40)
    #sensitivity = sum(sensitivity)/40

    #sens_loss = 1 - sensitivity

    #print(sensitivity, fps)

    return bce_loss, bce_loss, sensitivity, fps_image, froc_score, ground_truth_detections, predicted_detections


