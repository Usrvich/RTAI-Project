import torch
import torch.nn as nn
from DeepPolyVerifier import DeepPolyVerifier
import general_functions as gf

# Simple network similar to what we used in the lecture and exercises for efficient debugging of the depp poly verifier

type = "fc+lrelu" # fc or conv fc+relu
# conv and fc are passed
# relu and leaky relu are not passed
# type = "conv"
# type = "fc+relu"
input_dim = [1, 2, 2]

# fix the random seed for reproducibility
torch.manual_seed(0)
class CustomNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CustomNetwork, self).__init__()

        if type == "conv":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=[2,2], stride=[1,1], padding=[0,0])
            self.conv1.bias.data = self.conv1.bias.data * 0 + 1
            self.conv1.weight.data = torch.clamp(torch.round(self.conv1.weight.data*5), -1, 1)
        elif type == "fc":
            self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
            self.fc1 = nn.Linear(4,1)
            self.fc1.bias.data = self.fc1.bias.data * 0 + 1
            self.fc1.weight.data = torch.clamp(torch.round(self.fc1.weight.data*5), -1, 1)
        elif type == "fc+relu":
            self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
            self.fc1 = nn.Linear(4,1)
            self.fc1.bias.data = self.fc1.bias.data * 0 + 1
            self.fc1.weight.data = torch.clamp(torch.round(self.fc1.weight.data*5), -1, 1)
            self.relu = nn.ReLU()
        elif type == "fc+lrelu":
            self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
            self.fc1 = nn.Linear(4,1)
            self.fc1.bias.data = self.fc1.bias.data * 0 + 1
            self.fc1.weight.data = torch.clamp(torch.round(self.fc1.weight.data*5), -1, 1)
            self.relu = nn.LeakyReLU(negative_slope= 0.5)
        elif type == "fc+relu+fc+relu":
            self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
            self.fc1 = nn.Linear(4,3)
            self.fc1.bias.data = self.fc1.bias.data * 0 + 1
            self.fc1.weight.data = torch.clamp(torch.round(self.fc1.weight.data*5), -1, 1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(3,1)
            self.fc2.bias.data = self.fc2.bias.data * 0 + 1
            self.fc2.weight.data = torch.clamp(torch.round(self.fc2.weight.data*5), -1, 1)
            self.relu2 = nn.ReLU()
        else:
            raise NotImplementedError(f"Type {type} not implemented")

    def forward(self, x):
        if type == "conv":
            x = self.conv1(x)
        elif type == "fc":
            x = self.flatten(x)
            x = self.fc1(x)
        elif type == "fc+relu":
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
        elif type == "fc+lrelu":
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
        elif type == "fc+relu+fc+relu":
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
        return x


# creat am input_dim input tensor of ones
input = torch.ones(input_dim)
model = CustomNetwork(input_dim)

# print the model weights
print(model.state_dict())

eps = 1

print(f"Verify with input {input} and eps {eps}")
dpv = DeepPolyVerifier(model, input, eps, None, False, True)
optimizer = torch.optim.Adam(dpv.parameters(), lr=0.1)
input_upper_bound = input + eps
input_lower_bound = input - eps
for i in range(10):
    optimizer.zero_grad()
    out_lower_bounds, out_upper_bounds = dpv(dpv.input_lower_bound, dpv.input_upper_bound)
    loss = gf.overlap_error(None, out_lower_bounds, out_upper_bounds)
    print(loss)

    loss.backward()
    optimizer.step()

# get all combinations with +eps and -eps in each dim
combs = []
for i in range(2**input.nelement()):
    bin_mask = format(i, '0' + str(input.nelement()) + 'b')
    combination = input.clone()
    for j, bit in enumerate(bin_mask):
        if bit == '1':
            combination.view(-1)[j] += eps  # Add eps if the bit is 1
        else:
            combination.view(-1)[j] -= eps  # Subtract eps if the bit is 0
    combs.append(combination)

out = []
for comb in combs:
    out.append(model(comb))
out = torch.stack(out)
print(out.shape)
for i in range(out.shape[1]):
    print(f"Range for output neuron {i}: [{out[:,i].min()}, {out[:, i].max()}]")

for i in range(out.shape[1]):
    # compare the output range of the verifier with the output range computed above
    if (((dpv.out_upper_bounds[i] - out[:, i].min()).abs() > 1e-5) |
        ((dpv.out_upper_bounds[i]  - out[:, i].max()).abs() > 1e-5)):
        label = " --- NOT consistent"
    else:
        label = " --- ok"
    print(f"Range for output neuron{i}  manual/verifier : [{out[:, i].min():5.2f}, {out[:, i].max():5.2f}] / [{dpv.out_lower_bounds[i] :5.2f}, {dpv.out_upper_bounds[i] :5.2f}] {label}")
#
