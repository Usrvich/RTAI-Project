import numpy as np
import torch
from ConvTransformer import ConvTransformer
from LinearTransformer import LinearTransformer
from ReluTransformer import ReluTransformer
from InputTransformer import InputTransformer
from OutputTransformer import OutputTransformer


class DeepPolyVerifier(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, clamp=True,
                 verbose=False):
        super(DeepPolyVerifier, self).__init__()
        self.net = net
        self.eps = eps  # is the small perturbation of the input, trying to test the robustness of the network under
        # small perturbations
        self.true_label = true_label
        self.verbose = verbose
        self.netlist = []
        self.overlap = -np.inf
        self.transformer_list = torch.nn.ModuleList()

        self.out_lower_bounds = None
        self.out_upper_bounds = None

        # to make sure we can use the same code for images and vectors
        input_flat = inputs.reshape(-1).unsqueeze(0)
        if clamp:
            self.input_lower_bound = torch.clamp(input_flat - eps, min=0, max=1)
            self.input_upper_bound = torch.clamp(input_flat + eps, min=0, max=1)
        else:
            # we use this for the simple_network for debugging purposes. In the main code we should clamp!
            self.input_lower_bound = input_flat - eps
            self.input_upper_bound = input_flat + eps

        self.constraints = [{"weights_lower": self.input_lower_bound.unsqueeze(-1),
                             "weights_upper": self.input_upper_bound.unsqueeze(-1),
                             "bias_lower": torch.zeros_like(self.input_lower_bound).unsqueeze(-1),
                             "bias_upper": torch.zeros_like(self.input_lower_bound).unsqueeze(-1),
                             "bounds_lower": self.input_lower_bound,
                             "bounds_upper": self.input_upper_bound,
                             "shape_in": None,
                             "shape_out": torch.tensor(inputs.unsqueeze(0).shape)}]
        self.input_transformer = InputTransformer(self.constraints[0])
        self.transformer_list.append(self.input_transformer)
        prev_layer = None
        for param in self.net.parameters():
            param.requires_grad = False
        prev_transformer = self.input_transformer
        for name, layer in self.net.named_children():
            if isinstance(layer, torch.nn.Flatten):
                # we can ignore the flatten layer, because we work with flattened constraints
                continue
            elif isinstance(layer, torch.nn.Linear):

                self.netlist.append(layer)
                prev_transformer = LinearTransformer(prev_layer, layer, prev_transformer)
                self.transformer_list.append(prev_transformer)

            elif isinstance(layer, torch.nn.Conv2d):
                self.netlist.append(layer)
                prev_transformer = ConvTransformer(prev_layer, layer, prev_transformer)
                self.transformer_list.append(prev_transformer)

            elif isinstance(layer, torch.nn.ReLU) | isinstance(layer, torch.nn.LeakyReLU):
                self.netlist.append(layer)
                prev_transformer = ReluTransformer(prev_layer, layer, prev_transformer)
                self.transformer_list.append(prev_transformer)

            else:
                # should not happen when running evaluate, but just in case
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

            prev_layer = layer
        self.output_transformer = OutputTransformer(prev_layer, self.true_label, prev_transformer)
        self.transformer_list.append(self.output_transformer)

    def forward(self, lower_bound, upper_bound):
        lower_bound = self.input_lower_bound
        upper_bound = self.input_upper_bound
        for i, transformer in enumerate(self.transformer_list):
            if self.verbose and 0 < i < len(self.transformer_list) - 1:
                print(f"Layer {i - 1}: {self.netlist[i - 1]}")
                print(f"Input shape: {lower_bound.shape}")
                print(f"current transformer: {transformer}")
                print(f"lower bound before: {lower_bound}")
                print(f"upper bound before: {upper_bound}")
            lower_bound, upper_bound = transformer(lower_bound, upper_bound)
            if i > 0 and isinstance(self.transformer_list[i - 1], ReluTransformer):
                lower_bound, upper_bound = self.backsubstitute_1(i)
            if self.verbose and i > 0:
                print(f"lower bound after: {lower_bound}")
                print(f"upper bound after: {upper_bound}")

        old_out_lower_bound = self.transformer_list[-1].lower_bound.clone()
        old_out_upper_bound = self.transformer_list[-1].upper_bound.clone()

        lower_bound, upper_bound = self.backsubstitute_1(len(self.transformer_list) - 1, store_weights=True)

        new_lower_bounds = lower_bound
        new_upper_bounds = upper_bound

        self.out_lower_bounds = new_lower_bounds
        self.out_upper_bounds = new_upper_bounds

        self.transformer_list[-1].lower_bound = new_lower_bounds
        self.transformer_list[-1].upper_bound = new_upper_bounds

        if self.verbose:
            # show the new bounds after backsubstitution
            new_lower_bound = self.out_lower_bounds.clone().detach().reshape(-1)
            new_upper_bound = self.out_upper_bounds.clone().detach().reshape(-1)
            old_out_upper_bound = old_out_upper_bound.reshape(-1)
            old_out_lower_bound = old_out_lower_bound.reshape(-1)
            print(f"Bounds before/after backsubstitution:")
            for i in range(new_lower_bound.shape[0]):
                print(f" [{old_out_lower_bound[i - 1]:7.4f},  {old_out_upper_bound[i - 1]:7.4f}]  -->   "
                      f"[{new_lower_bound[i - 1]:7.4f},  {new_upper_bound[i - 1]:7.4f}]")
            print(f"true label: {self.true_label}")
            print(f"predicted label: {torch.argmax(self.out_upper_bounds)}")

        return self.out_lower_bounds, self.out_upper_bounds

    def backsubstitute_1(self, cur_idx, store_weights=False):

        lc_cur_b = self.transformer_list[cur_idx].bias_lower.clone().squeeze(0)  # shape: dimL(last) x 1
        uc_cur_b = self.transformer_list[cur_idx].bias_upper.clone().squeeze(0)  # shape: dimL(last) x 1
        lc_cur_w = self.transformer_list[cur_idx].weights_lower.clone().squeeze(0)  # shape: dimL(last) x dimL(prev)
        uc_cur_w = self.transformer_list[cur_idx].weights_upper.clone().squeeze(0)  # shape: dimL(last) x dimL(prev)

        for i in range(cur_idx - 1, -1, -1):
            # get the constraints from the previous layer
            lc_prev_b = self.transformer_list[i].bias_lower.clone().squeeze(0)  # shape: 1 x dimL(i) x 1
            uc_prev_b = self.transformer_list[i].bias_upper.clone().squeeze(0)
            lc_prev_w = self.transformer_list[i].weights_lower.clone().squeeze(0)  # shape: 1 x dimL(i) x dimL(i-1)
            uc_prev_w = self.transformer_list[i].weights_upper.clone().squeeze(0)

            pivot_matrix = torch.zeros_like(lc_cur_w)
            lc_cur_w_positive = torch.max(lc_cur_w, pivot_matrix)
            lc_cur_w_negative = torch.min(lc_cur_w, pivot_matrix)
            lc_cur_w = lc_cur_w_positive @ lc_prev_w + lc_cur_w_negative @ uc_prev_w
            lc_cur_b = lc_cur_w_positive @ lc_prev_b + lc_cur_w_negative @ uc_prev_b + lc_cur_b

            uc_cur_w_positive = torch.max(uc_cur_w, pivot_matrix)
            uc_cur_w_negative = torch.min(uc_cur_w, pivot_matrix)
            uc_cur_w = uc_cur_w_positive @ uc_prev_w + uc_cur_w_negative @ lc_prev_w
            uc_cur_b = uc_cur_w_positive @ uc_prev_b + uc_cur_w_negative @ lc_prev_b + uc_cur_b

            if store_weights & (i == 1):
                self.constraints_first_to_final = {}
                self.constraints_first_to_final["weights_lower"] = lc_cur_w
                self.constraints_first_to_final["weights_upper"] = uc_cur_w
                self.constraints_first_to_final["bias_lower"] = lc_cur_b
                self.constraints_first_to_final["bias_upper"] = uc_cur_b

        new_lower_bound = lc_cur_w + lc_cur_b
        new_upper_bound = uc_cur_w + uc_cur_b
        self.transformer_list[cur_idx].lower_bound = new_lower_bound
        self.transformer_list[cur_idx].upper_bound = new_upper_bound
        return new_lower_bound, new_upper_bound


    def linear_transform(self, prev_transformer, prev_layer, cur_layer) -> None:

        linear_transformer = LinearTransformer(prev_transformer, prev_layer, cur_layer)
        linear_transformer()

        self.transformer_list.append(linear_transformer)
        self.constraints.append({"weights_lower": linear_transformer.weights_lower,
                                 "weights_upper": linear_transformer.weights_upper,
                                 "bias_lower": linear_transformer.bias_lower,
                                 "bias_upper": linear_transformer.bias_upper,
                                 "bounds_lower": linear_transformer.lower_bound,
                                 "bounds_upper": linear_transformer.upper_bound,
                                 "shape_in": linear_transformer.shape_in,
                                 "shape_out": linear_transformer.shape_out})

    def conv2d_transform(self, prev_transformer, prev_layer, cur_layer) -> None:
        conv_transformer = ConvTransformer(prev_transformer, prev_layer, cur_layer)
        conv_transformer()
        self.transformer_list.append(conv_transformer)
        self.constraints.append({"weights_lower": conv_transformer.weights_lower,
                                 "weights_upper": conv_transformer.weights_upper,
                                 "bias_lower": conv_transformer.bias_lower,
                                 "bias_upper": conv_transformer.bias_upper,
                                 "bounds_lower": conv_transformer.lower_bound,
                                 "bounds_upper": conv_transformer.upper_bound,
                                 "shape_in": conv_transformer.shape_in,
                                 "shape_out": conv_transformer.shape_out})

    def relu_transform_adjustable(self, prev_transformer, prev_layer, cur_layer) -> None:
        relu_transformer = ReluTransformer(prev_transformer, prev_layer, cur_layer)
        relu_transformer()
        self.transformer_list.append(relu_transformer)
        self.constraints.append({"weights_lower": relu_transformer.weights_lower,
                                 "weights_upper": relu_transformer.weights_upper,
                                 "bias_lower": relu_transformer.bias_lower,
                                 "bias_upper": relu_transformer.bias_upper,
                                 "bounds_lower": relu_transformer.lower_bound,
                                 "bounds_upper": relu_transformer.upper_bound,
                                 "shape_in": relu_transformer.shape_in,
                                 "shape_out": relu_transformer.shape_out})

    # deepPoly relaxation
    def compute_bounds(self, weights_lower, weights_upper, bias_lower, bias_upper, lower_bounds, upper_bounds):
        # weights should have dim: 1 x output_flat x input_flat
        # bias should have dim: 1 x output_flat x 1

        lower_bounds_ = torch.where(weights_lower > 0, lower_bounds, upper_bounds)
        upper_bounds_ = torch.where(weights_upper > 0, upper_bounds, lower_bounds)

        # print("shapes: ",lower_bounds_.shape, weights_lower.shape, bias_lower.shape)

        # compute the new lower/upper bounds for the next layer
        # in the forward case

        lower_bound_new = (lower_bounds_ * weights_lower).sum(dim=2) + bias_lower.reshape(1, -1)
        upper_bound_new = (upper_bounds_ * weights_upper).sum(dim=2) + bias_upper.reshape(1, -1)
        return lower_bound_new, upper_bound_new

    def check_postcondition_p(self) -> bool:
        target = self.true_label
        # target_lb = self.out_lower_bounds[target]
        provings = 0
        for i in range(self.out_upper_bounds.shape[0]):
            if i != target and self.out_lower_bounds[i] < 0:
                # via proving robustness
                lc_prev_w = self.transformer_list[0].weights_lower
                uc_prev_w = self.transformer_list[0].weights_upper
                weight_diff = self.constraints_first_to_final["weights_lower"][target, :] - \
                              self.constraints_first_to_final["weights_upper"][i, :]
                bias_diff = self.constraints_first_to_final["bias_lower"][target, :] - self.constraints_first_to_final[
                                                                                           "bias_upper"][i, :]
                diff = torch.clamp(weight_diff, min=0).reshape(1, -1) @ lc_prev_w.squeeze(0) + \
                       torch.clamp(weight_diff, max=0).reshape(1, -1) @ uc_prev_w.squeeze(0) + \
                       bias_diff.reshape(1, -1)
                if diff <= 0:
                    return False
                else:
                    provings += 1
        if provings > 0:
            print(f"via proving robustness of {provings} classes")
        return True

    # def check_postcondition(self) -> bool:
    #     target = self.true_label
    #     # target_lb = self.out_lower_bounds[target]
    #     # for i in range(self.out_upper_bounds.shape[0]):
    #     #     if i != target and self.out_upper_bounds[i] >= target_lb:
    #     #         return False
    #     # return True
    #     if torch.all(self.out_lower_bounds >= 0):
    #         return True
    #     else:
    #         return False

    def check_postcondition(self, loss) -> bool:
        if loss <= 0:
            return True
        else:
            return False