import os
from utils.loading import parse_spec
from networks import get_network
from verifier import analyze
import numpy as np
import torch

# This script checks all networks and all specs in the test_cases folder
# It compares the predicted status with the gt.txt file and prints the inconsistencies
# This code runs much faster than the evaluate script and is therefore useful for quickly checking if a new implementation broke another setting

DEVICE = "cpu"
networks = ["fc_base", "fc_1", "fc_2", "fc_3", "fc_4", "fc_5", "fc_6", "fc_7", "conv_base", "conv_1", "conv_2", "conv_3", "conv_4"]
for folder in ["test_cases", "preliminary_evaluation_test_cases"]:
    print(f"\n------------{folder}------------------------")
    print(f"{'network':10} {'spec':25}  {'predict':15} {'ground truth':15} {'points':6} {'MinLoss':7} {'Time':6} {'Iter':6} {'Relu':5} {'LrSlopeMin':10} {'LrSlopeMax':10}")
    points_total = 0
    possible_points_total = 0
    for net_name in networks:
        # List specification files with path under each network folder
        if not os.path.exists(f"../{folder}/{net_name}"):
            continue
        spec_files = os.listdir(f"../{folder}/{net_name}")
        spec_files = np.sort([f"../{folder}/{net_name}/{spec}" for spec in spec_files])
        for spec_idx in range(len(spec_files)):
            spec = spec_files[spec_idx]

            true_label, dataset, image, eps = parse_spec(spec)

            net = get_network(net_name, dataset, f"../models/{dataset}_{net_name}.pt").to(DEVICE)
            has_relu = np.any([isinstance(m, torch.nn.ReLU) for m in net.modules()])
            has_lrelu = np.any([isinstance(m, torch.nn.LeakyReLU) for m in net.modules()])
            if has_lrelu:
                lrelu_slope = [m.negative_slope for m in net.modules() if isinstance(m, torch.nn.LeakyReLU)]
                lrelu_slope_min = torch.min(torch.tensor(lrelu_slope))
                lrelu_slope_max = torch.max(torch.tensor(lrelu_slope))
            else:
                lrelu_slope_min = np.nan
                lrelu_slope_max = np.nan
            # print(net)
            # print(f"ReLU: {lrelu_slope_min_max}")
            # break

            image = image.to(DEVICE)
            out = net(image.unsqueeze(0))

            pred_label = out.max(dim=1)[1].item()
            assert pred_label == true_label

            # get the gt label from gt.txt as str to compare with the predicted label
            gt = np.loadtxt(f"../{folder}/gt.txt", dtype=str, delimiter=',')
            gt_dict = {}
            for model, img, status in gt:
                if model not in gt_dict:
                    gt_dict[model] = {}
                gt_dict[model][img] = status

            postcondition, time_elapsed, iterations, loss = analyze(net, image, eps, true_label, return_metrics=True)
            if postcondition:
                predict_status = "verified"
            else:
                predict_status = "not verified"

            if gt_dict[net_name][spec.split('/')[-1]] == predict_status:
                if predict_status == "verified":
                    points = 1
                else:
                    points = 0
            else:
                if predict_status == "verified":
                    points = -2
                else:
                    points = 0
            if gt_dict[net_name][spec.split('/')[-1]] == "verified":
                possible_points = 1
            else:
                possible_points = 0
            print(f"{net_name:10} {spec.split('/')[-1]:25} {predict_status:15} {gt_dict[net_name][spec.split('/')[-1]]:15} {points:4}/{possible_points:1} "
                 f"{loss:7.5f} {time_elapsed:5.1f}s {iterations:6}"
                  f"{has_relu:5} {lrelu_slope_min:10.8} {lrelu_slope_max:10.8}")
            points_total += points
            possible_points_total += possible_points
    print(f"Total points: {points_total}/{possible_points_total}")

