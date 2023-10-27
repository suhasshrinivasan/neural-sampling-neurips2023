import math

import gensn.distributions as G
import torch
import torch.distributions as D
from gensn.distributions import TrainableDistributionAdapter
from gensn.flow import FlowDistribution
from gensn.transforms.invertible import (
    ELU,
    ConditionalShift,
    ELUplus1,
    Exp,
    IndependentAffine,
    InverseTransform,
    Log,
    Pow,
    SequentialTransform,
    Sigmoid,
    Softplus,
    Sqrt,
    Tanh,
)
from torch import nn

from neural_sampling_code.elements.conditioned_parameters import (
    CondMeanMLP,
    CondScaleMLPExp,
)


def initialize_affine_transform_(affine, init_mean, init_std=None):
    if init_std is None:
        init_std = 1.0 / math.sqrt(affine.weight.shape[0])
    affine.weight.data.normal_(mean=init_mean, std=init_std)
    # TODO: reconsider this initialization
    affine.bias.data.normal_(mean=init_mean, std=init_std)


nonlin_lookup = {
    "elu": ELU,
    "inv_elu": lambda: InverseTransform(ELU()),
    "tanh": Tanh,
    "inv_tanh": lambda: InverseTransform(Tanh()),
    "sigmoid": Sigmoid,
    "inv_sigmoid": lambda: InverseTransform(Sigmoid()),
    "exp": Exp,
    "inv_exp": lambda: InverseTransform(Exp()),
    "softplus": Softplus,
    "inv_softplus": lambda: InverseTransform(Softplus()),
    "eluplus1": ELUplus1,
    "inv_eluplus1": lambda: InverseTransform(ELUplus1()),
    "log": Log,
    "inv_log": lambda: InverseTransform(Log()),
    "pow2": lambda: Pow(2),
    "inv_pow2": lambda: InverseTransform(Pow(2)),
    "pow3": lambda: Pow(3),
    "inv_pow3": lambda: InverseTransform(Pow(3)),
    "sqrt": Sqrt,
    "inv_sqrt": lambda: InverseTransform(Sqrt()),
}


def build_transform_sequence(
    dims=1,
    depth=5,
    nonlin="elu",
    initial_nonlin="none",
    final_nonlin="none",
    affine_init_mean=1.3,
    affine_init_std=None,
):
    layers = []

    # put the first nonlinearity if given
    if initial_nonlin != "none":
        layers.append(nonlin_lookup[initial_nonlin]())

    # stack sequence of affine + nonlin
    # NOTE: this assumes that sharing identical nonlin layer
    # multiple times in a network is okay. Probably fine.
    nonlin_layer = nonlin_lookup[nonlin]()
    for _ in range(depth):
        affine_layer = IndependentAffine(dims)
        initialize_affine_transform_(
            affine_layer, init_mean=affine_init_mean, init_std=affine_init_std
        )
        layers.append(affine_layer)
        layers.append(nonlin_layer)

    # why is this separated?
    if depth > 0:
        affine_layer = IndependentAffine(dims)
        initialize_affine_transform_(
            affine_layer, init_mean=affine_init_mean, init_std=affine_init_std
        )
        layers.append(affine_layer)

    #  put final nonlinearity if given
    if final_nonlin != "none":
        layers.append(nonlin_lookup[final_nonlin]())

    return layers


def get_flow_prior_model(
    dims,
    flow_base_distribution,
    flow_depth,
    flow_initial_nonlinearity,
    flow_nonlinearity,
):
    if flow_base_distribution == "normal":
        flow_base_distribution = G.IndependentNormal(
            loc=torch.zeros(dims), scale=torch.ones(dims)
        )
    elif flow_base_distribution == "uniform":
        flow_base_distribution = TrainableDistributionAdapter(
            G.wrap_with_indep(D.Uniform), low=torch.zeros(dims), high=torch.ones(dims)
        )
    # TODO: add cases for conditional normal / uniform by simple construction
    else:
        raise ValueError("Unknown base distribution for flow")

    # create the sequential flow
    flow_transform = SequentialTransform(
        *build_transform_sequence(
            dims=dims,
            depth=flow_depth,
            nonlin=flow_nonlinearity,
            initial_nonlin=flow_initial_nonlinearity,
        )
    )

    # construct the flow distribution
    return FlowDistribution(flow_base_distribution, flow_transform)


# TODO: merge with get_latent_density_model
def get_flow_dequantizer(
    dims,
    flow_base_distribution,
    flow_depth,
    flow_initial_nonlinearity,
    flow_nonlinearity,
):
    if flow_base_distribution == "normal":
        flow_base_distribution = TrainableDistributionAdapter(
            D.Normal, loc=torch.zeros(dims), scale=torch.ones(dims)
        )
    elif flow_base_distribution == "uniform":
        flow_base_distribution = TrainableDistributionAdapter(
            D.Uniform, low=torch.zeros(dims), high=torch.ones(dims)
        )
    elif flow_base_distribution == "conditional_normal":
        flow_base_distribution = G.IndependentNormal(
            loc=CondMeanMLP(in_features=dims, out_features=dims),
            scale=CondScaleMLPExp(in_features=dims, out_features=dims),
        )
    # TODO: add cases for conditional uniform by simple construction
    else:
        raise ValueError("Unknown base distribution for flow")

    # create the sequential flow
    flow_sequence = build_transform_sequence(
        dims=dims,
        depth=flow_depth,
        nonlin=flow_nonlinearity,
        initial_nonlin=flow_initial_nonlinearity,
    )

    # prepare layer to remove x from z
    deshift_transform = InverseTransform(ConditionalShift(nn.Identity()))

    # always start with deshifting layer:
    flow_sequence.insert(0, deshift_transform)

    flow_transform = SequentialTransform(*flow_sequence)

    return FlowDistribution(flow_base_distribution, flow_transform)


# def get_dequantization_trainer(optimizer, lr,)
