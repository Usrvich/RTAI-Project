import argparse

import numpy as np
import torch

from networks import get_network
from utils.loading import parse_spec

from DeepPolyVerifier import DeepPolyVerifier
from ReluTransformer import ReluTransformer

import general_functions as gf
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.autograd import gradcheck
import time

DEVICE = "cpu"


def analyze(
        net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, return_metrics=False
) -> bool:
    dpv = DeepPolyVerifier(net, inputs, eps, true_label, verbose=False)

    has_relu = False
    loss = np.inf
    for name, param in dpv.transformer_list.named_parameters():
        if not param.requires_grad:
            continue
        # print(name, param.data)
        has_relu = True

    # unverified if out of time

    start_time = time.time()
    # loss_track = []

    count = 0
    loss_track = [1e6]

    # show the network structure
    # print("network structure:")
    # for layer in dpv.netlist:
    #     print(layer)

    if has_relu:
        reinit_after = 10  # to escape local minima
        limit_train = 20000
        limit = limit_train
        verify = False
        count_reinit = 0
        num_reinits = 0
        while not verify and count < limit:
            cur_time = time.time()
            if cur_time - start_time > 60:
                break

            loss_movavg = np.mean(loss_track[-reinit_after:])
            # reinitialize the alphas and change the loss function
            if ((count == 0) |
                    ((count > count_reinit + reinit_after) & (loss_track[-1]/loss_movavg>1-1e-3))):
                count_reinit = count
                # print("reinit")

                # optimizer = torch.optim.AdamW(dpv.parameters(), lr=100, weight_decay=1e-4)
                optimizer = torch.optim.SGD(dpv.parameters(), lr=9, momentum=0.9)
                scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
                # scheduler = ExponentialLR(optimizer, gamma=0.9)

                for j in range(len(dpv.transformer_list)):
                    if isinstance(dpv.transformer_list[j], ReluTransformer):
                        # print("resetting alpha")
                        dpv.transformer_list[j].alpha_initialized = False
                if num_reinits % 5 == 0:
                    num_reinits += 1
                    loss_func = gf.LossFunction("mae")
                elif num_reinits % 5 == 1:
                    num_reinits += 1
                    loss_func = gf.LossFunction("overlap_huber", delta=.005)
                    # print("reinit loss func to huber with delta=.005")
                elif num_reinits % 5 == 2:
                    num_reinits += 1
                    loss_func = gf.LossFunction("mse")
                    # print("reinit loss func to MSE")
                elif num_reinits % 5 == 3:
                    num_reinits += 1
                    loss_func = gf.LossFunction("overlap_smooth")
                elif num_reinits % 5 == 4:
                    num_reinits += 1
                    loss_func = gf.LossFunction("overlap_huber", delta=.05)
                    # print("reinit loss func to huber with delta=.05")
                count += 1
                continue

            elif count > limit_train: # brute force
                loss_fcn_label = "brute_force"
                for j in range(len(dpv.transformer_list)):
                    if isinstance(dpv.transformer_list[j], ReluTransformer):
                        dpv.transformer_list[j].alpha_initialized = False
                        # dpv.transformer_list[j].alpha.data = torch.rand_like(dpv.transformer_list[j].alpha.data)
                out_lower_bounds, out_upper_bounds = dpv(dpv.input_lower_bound, dpv.input_upper_bound)
                loss = loss_func(out_lower_bounds, dpv.transformer_list)

            else:
                optimizer.zero_grad()
                out_lower_bounds, out_upper_bounds = dpv(dpv.input_lower_bound, dpv.input_upper_bound)
                # loss = gf.overlap_error(out_lower_bounds)
                loss = loss_func(out_lower_bounds, dpv.transformer_list)

                loss_track = np.append(loss_track, loss.item())
                loss.backward()
                # check the gradients
                # for name, param in dpv.named_parameters():
                #     if param.requires_grad:
                #         # find the smallest x% of the gradients and add noise to them
                #         # the longer we train, the larger the noise
                #         threshold_perc = .5
                #         flat_grad = param.grad.flatten()
                #         lowk_indices = torch.argsort(torch.abs(flat_grad))[:int(threshold_perc * len(flat_grad))]
                #         flat_grad[lowk_indices] += torch.rand_like(flat_grad[lowk_indices]) * \
                #                                    torch.logspace(-7, -3, limit)[count]
                #         param.grad = flat_grad.view_as(param.grad)
                        # print(f"param: {name} / grad: {torch.norm(param.grad)}")
                optimizer.step()
                scheduler.step()



                # # add noise to the alpha values
                # for j in range(len(dpv.transformer_list)):
                #     if isinstance(dpv.transformer_list[j], ReluTransformer):
                #         dpv.transformer_list[j].alpha.data += torch.rand_like(dpv.transformer_list[j].alpha.data) * torch.linspace(0, .5, limit)[count]

            count += 1
            verify = dpv.check_postcondition_p()
            # verify = dpv.check_postcondition(loss)

            # print(f" {count:04d} / {time.time() - start_time:.0f}s / loss: {loss.item():.4f} / loss_movavg: {loss_movavg:.4f} / loss_fcn: {loss_func.loss_fcn_label}")
    else:
        out_lower_bounds, out_upper_bounds = dpv(dpv.input_lower_bound, dpv.input_upper_bound)
        loss = gf.overlap_error(out_lower_bounds)
        loss_track = np.append(loss_track, loss.item())
    if return_metrics:
        return dpv.check_postcondition_p(), time.time() - start_time, count, np.min(loss_track)
    else:
        return dpv.check_postcondition_p()

def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
