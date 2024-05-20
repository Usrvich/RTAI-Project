import os
from utils.loading import parse_spec
from networks import get_network
from verifier import analyze
import numpy as np

# This script checks a specified network and spec in the test_cases folder
# It compares the predicted status with the gt.txt file and prints the inconsistencies


# define which net and spec to check
net_name = "conv_4"
spec_idx = 0

DEVICE = "cpu"
networks = ["fc_base", "fc_1", "fc_2", "fc_3", "fc_4", "fc_5", "fc_6", "fc_7", "conv_base", "conv_1", "conv_2", "conv_3", "conv_4"]
folder = ["test_cases", "preliminary_evaluation_test_cases"][0]

assert net_name in networks, f"Network {net_name} not found in {networks}"

# List specification files with path under each network folder
spec_files = os.listdir(f"../preliminary_evaluation_test_cases/{net_name}")
spec_files = np.sort([f"../preliminary_evaluation_test_cases/{net_name}/{spec}" for spec in spec_files])

spec = spec_files[spec_idx]

true_label, dataset, image, eps = parse_spec(spec)

net = get_network(net_name, dataset, f"../models/{dataset}_{net_name}.pt").to(DEVICE)

image = image.to(DEVICE)
out = net(image.unsqueeze(0))

pred_label = out.max(dim=1)[1].item()
assert pred_label == true_label

# get the gt label from gt.txt as str to compare with the predicted label
gt = np.loadtxt(f"../preliminary_evaluation_test_cases/gt.txt", dtype=str, delimiter=',')
gt_dict = {}
for model, img, status in gt:
    if model not in gt_dict:
        gt_dict[model] = {}
    gt_dict[model][img] = status

if analyze(net, image, eps, true_label):
    predict_status = "verified"
else:
    predict_status = "not verified"

if gt_dict[net_name][spec.split('/')[-1]] == predict_status:
    check_label = "ok"
else:
    check_label = "------ INCONSISTENT"
print(f"{'network':10} {'spec':25}  {'predict':15} {'ground truth':15} {'result':10}")
print(f"{net_name:10} {spec.split('/')[-1]:25} {predict_status:15} {gt_dict[net_name][spec.split('/')[-1]]:15} {check_label:10}")

