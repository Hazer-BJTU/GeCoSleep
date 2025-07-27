import torch
import torch.nn.functional as F

##############################################################################
################ implemenations of the basic functionality ###################
##############################################################################

def error(y, pred):
    return torch.mean(torch.ne(y, pred))


def accuracy(y, pred):
    return torch.mean(torch.eq(y, pred))


def clip(x, min, max):
    return torch.clamp(x, min, max)


def floor(x):
    return torch.floor(x).int()


def ceil(x):
    return torch.ceil(x).int()


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return F.relu(x)


def leaky_relu(x, negative_slope):
    return F.leaky_relu(x, negative_slope=negative_slope)


def softplus(x):
    return F.softplus(x)


def softmax(x):
    return F.softmax(x)


def tanh(x):
    return torch.tanh(x)


def l2_norm(x, epsilon=0.00001):
    square_sum = torch.sum(torch.pow(x, exponent=2))
    norm = torch.sqrt(torch.add(square_sum, epsilon))
    return norm


def l2_norm_2d(x, epsilon=0.00001):
    square_sum = torch.sum(torch.pow(x, exponent=2))
    norm = torch.mean(torch.sqrt(torch.add(square_sum, epsilon)))
    return norm


# we assume afa=beta
def neg_likelihood_gamma(x, afa, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    neg_likelihood = -(afa - 1) * torch.log(norm) + afa * norm
    return torch.mean(neg_likelihood)


# KL(lambda_t||lambda=1)
def kl_exponential(x, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    kl = -torch.log(norm) + norm
    return torch.mean(kl)


def likelihood(x, y, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    kl = -torch.log(norm) + norm * y
    return 0.25 * torch.mean(kl)


def shape(x):
    return x.shape


def reshape(x, shape):
    y = torch.reshape(x, shape).float()
    return y


def Linear_Function(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret

    