import math

import torch
from torch import nn


# TODO: gather these kinds of standard initializers in its own
# package/module for better organization. This need not be
# just specific for usage here.
def prepare_init(init_std=1e-3, init_bias=0.0):
    """
    Returns an initializer applicable for initializing
    nn.Linear.

    Specifically, nn.Linear is initialized as follows:
    * nn.init_normal_(weight, std=init_std)
    * nn.init.constant_(bias, 0.0)
    """

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=init_std)
            nn.init.constant_(m.bias, init_bias)

    return _init_weights


class MLPCore(nn.Module):
    """
    A general sequential MLP module that can be used as a
    core module before applying readouts.
    """

    def __init__(
        self,
        in_features,
        out_features,
        n_layers=1,
        nonlinearity="relu",
        dropout_rate=0.0,
        init_std=1e-3,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        # TODO: different dropout rates for different layers?
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        # TODO: what's the right number of layers for the hidden layer?
        # there isn't a good answer in general
        self.hidden_features = (in_features + out_features) // 2
        self.nonlinearity = nonlinearity
        # build the sequential MLP module
        self.core_module = nn.Sequential()
        self.core_module.add_module(
            "linear_0", nn.Linear(in_features, self.hidden_features)
        )
        # add dropout after each nn.Linear
        self.core_module.add_module("dropout_0", nn.Dropout(dropout_rate))
        self.core_module.add_module(
            "nonlinearity_0", nonlinearity_lookup[nonlinearity]()
        )
        for i in range(1, n_layers):
            self.core_module.add_module(
                f"linear_{i}", nn.Linear(self.hidden_features, self.hidden_features)
            )
            self.core_module.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            self.core_module.add_module(
                f"nonlinearity_{i}", nonlinearity_lookup[nonlinearity]()
            )

        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        return self.core_module(x)


# Take the log scale parameter from below and create a
# new class that just gives a CondScale
class CondScaleMLPExp(nn.Module):
    def __init__(self, in_features, out_features, init_std=1e-3):
        super().__init__()
        self.log_scale_module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.init_std = init_std
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        scale = self.log_scale_module(x).exp()
        finfo = torch.finfo(scale.dtype)
        scale = scale + finfo.tiny
        return scale


# Do the same for the mean
class CondMeanMLP(nn.Module):
    def __init__(self, in_features, out_features, init_std=1e-3):
        super().__init__()
        self.mean_module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.init_std = init_std
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        return self.mean_module(x)


nonlinearity_lookup = {
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "relu": torch.nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}


class MeanStdMLPExp(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity="relu", init_std=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = (in_features + out_features) // 2
        self.core_module = nn.Sequential(
            nn.Linear(in_features, self.hidden_features),
            nonlinearity_lookup[nonlinearity](),
        )
        self.mean_module = nn.Linear(self.hidden_features, out_features)
        self.log_scale_module = nn.Linear(self.hidden_features, out_features)
        self.init_std = init_std
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        core_out = self.core_module(x)
        mean = self.mean_module(core_out)
        scale = self.log_scale_module(core_out).exp()
        finfo = torch.finfo(scale.dtype)
        scale = scale + finfo.tiny
        return mean, scale


nonneg_transform_lookup = {
    "exp": torch.exp,
    "softplus": torch.nn.functional.softplus,
    "relu": torch.relu,
    "eluplus1": lambda x: torch.nn.functional.elu(x) + 1,
    "abs": torch.abs,
    "sq": lambda x: x**2,
    "sigmoid": torch.sigmoid,
    "none": lambda x: x,
}


class MeanStdMLPFlexible(nn.Module):
    """
    A flexible MLP that can be used to model the mean and standard deviation
    of a distribution. The mean and standard deviation are modeled separately
    by two MLPs. The standard deviation is modeled by a non-negative MLP
    that is transformed by a non-negative function. The mean is modeled by
    an MLP that is not constrained to be non-negative.
    """

    # default values for the parameters
    # NOTE: this was done in addition to adding the parameters to the
    # init in order to support backwards compatibility of models saved
    # before these parameters were added
    # THIS IS A HACK!
    clamp_pre_std = True
    pre_std_min = -8
    pre_std_max = 10
    # TODO: change these back to the above values
    # pre_std_min = -1
    # pre_std_max = 1

    def __init__(
        self,
        in_features,
        out_features,
        n_layers=2,
        nonlinearity="relu",
        dropout_rate=0.0,
        nonneg_std_transform="exp",
        clamp_pre_std=True,
        pre_std_min=-8,
        pre_std_max=10,
        init_std=1e-3,
    ):
        """
        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            n_layers (int): number of layers in the MLP
            nonlinearity (str): nonlinearity to use in the MLP
            dropout_rate (float): dropout rate to use in the MLP
            nonneg_std_transform (str): non-negative transform to apply to the
                output of the standard deviation MLP
            clamp_pre_std (bool): whether to clamp the output of the standard
                deviation MLP to be in the range [pre_std_min, pre_std_max]
            pre_std_min (float): minimum value of the output of the standard
                deviation MLP before the non-negative transform
            pre_std_max (float): maximum value of the output of the standard
                deviation MLP before the non-negative transform
            init_std (float): standard deviation of the initialization

        Remarks:
            The standard deviation MLP is constrained to be non-negative
            by applying a non-negative transform to its output. The
            non-negative transform is applied to the output of the MLP
            before the standard deviation is computed.
            The default values given to pre_std_min and pre_std_max
            are chosen to be reasonable for the "exp" transform, and
            specific to the domain of images that are normalized to be
            in the range [0, 1].
            clamp_pre_std is set to True by default, which means that
            the output of the standard deviation MLP is clamped to be
            in the range [pre_std_min, pre_std_max].
        """
        super().__init__()
        # supports only >= 2 layers since it's an MLP
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        self.in_features = in_features
        self.out_features = out_features
        # TODO: different dropout rates for different layers?
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        self.nonneg_std_transform = nonneg_std_transform
        self.clamp_pre_std = clamp_pre_std
        self.pre_std_min = pre_std_min
        self.pre_std_max = pre_std_max
        # TODO: what's the right number of layers for the hidden layer?
        # there isn't a good answer in general
        self.hidden_features = (in_features + out_features) // 2
        self.nonlinearity = nonlinearity
        # build the sequential MLP module
        self.core_module = nn.Sequential()
        self.core_module.add_module(
            "linear_0", nn.Linear(in_features, self.hidden_features)
        )
        # add dropout after each nn.Linear
        self.core_module.add_module("dropout_0", nn.Dropout(dropout_rate))
        self.core_module.add_module(
            "nonlinearity_0", nonlinearity_lookup[nonlinearity]()
        )
        for i in range(1, n_layers - 1):
            self.core_module.add_module(
                f"linear_{i}", nn.Linear(self.hidden_features, self.hidden_features)
            )
            self.core_module.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            self.core_module.add_module(
                f"nonlinearity_{i}", nonlinearity_lookup[nonlinearity]()
            )
        # add the final mean and log scale readout layers
        self.mean_module = nn.Linear(self.hidden_features, out_features)
        self.pre_std_module = nn.Linear(self.hidden_features, out_features)
        # initialize the weights
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        core_out = self.core_module(x)
        mean = self.mean_module(core_out)
        pre_std = self.pre_std_module(core_out)
        # apply the nonzero transform to the pre_std to get std
        if self.clamp_pre_std:
            pre_std = pre_std.clamp(min=self.pre_std_min, max=self.pre_std_max)
        std = nonneg_transform_lookup[self.nonneg_std_transform](pre_std)
        return mean, std


class ConcRateMLPExp(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity="relu", init_std=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = (in_features + out_features) // 2
        self.core_module = nn.Sequential(
            nn.Linear(in_features, self.hidden_features),
            nonlinearity_lookup[nonlinearity](),
        )
        self.log_conc_module = nn.Linear(self.hidden_features, out_features)
        self.log_rate_module = nn.Linear(self.hidden_features, out_features)
        self.init_std = init_std
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        core_out = self.core_module(x)
        conc = self.log_conc_module(core_out).exp()
        rate = self.log_rate_module(core_out).exp()
        finfo = torch.finfo(rate.dtype)
        conc = conc + finfo.tiny
        rate = rate + finfo.tiny
        return conc, rate


# NOTE: unlike MeanStdMLPFlexible, this uses the MLPCore module
# This is a PyTorch module for a flexible MLP-based concentration rate model with customizable number
# of layers, nonlinearity, dropout rate, initialization, and non-negative transformation.
class ConcRateMLPFlexible(nn.Module):
    """
    A PyTorch module for a flexible MLP-based concentration-rate model with customizable number
    of layers, nonlinearity, dropout rate, initialization, and non-negative transformation.
    This is meant to used to parameterize the concentration and rate of a Gamma distribution.
    """

    def __init__(
        self,
        in_features,
        out_features,
        n_layers=2,
        nonlinearity="relu",
        dropout_rate=0.0,
        init_std=1e-3,
        nonneg_transform="exp",
        clamp_pre_conc=True,
        pre_conc_max=2.0,
        clamp_pre_rate=True,
        pre_rate_min=-1.6,
    ):
        """
        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            n_layers (int): number of layers in the MLP
            nonlinearity (str): nonlinearity to use in the MLP
            dropout_rate (float): dropout rate to use in the MLP
            init_std (float): standard deviation to use for initialization
            nonneg_transform (str): non-negative transform to use for the log conc and log rate
            clamp_pre_conc (bool): whether to clamp the pre-conc values
            pre_conc_max (float): maximum value for the pre-conc values
            clamp_pre_rate (bool): whether to clamp the pre-rate values
            pre_rate_min (float): minimum value for the pre-rate values

        Remarks:
            - The total number of layers in the MLP is n_layers
            - The first n_layers - 1 layers are the core layers
            - The last layer is the log conc and log rate readout layer
            - The non-negative transform is applied to the log conc and log rate
            - The pre_conc_max is used to down-clamp the pre-conc values if clamp_pre_conc is True
            - The default value for pre_conc_max is 4.0, is chosen such that the exp transform
                on pre_conc_max is exp(4.0) = 54.6, which is close to the maximum value of
                neuronal responses observed in the dataset as this module is generally used
                for modeling response distribution
            - Similarly, we also do an up-clamp on the pre-rate values such that the rate
                values are >= 0.2 (which would alread be quite a wide gamma distribution), hence
                default value of pre_rate_min is -1.6 (log(0.2) = -1.6)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        self.nonneg_transform = nonneg_transform
        self.clamp_pre_conc = clamp_pre_conc
        self.pre_conc_max = pre_conc_max
        self.clamp_pre_rate = clamp_pre_rate
        self.pre_rate_min = pre_rate_min
        # build the sequential MLP module
        # subtract 1 layer from n_layers to form n_core_layers
        self.n_core_layers = self.n_layers - 1
        self.core_module = MLPCore(
            in_features=in_features,
            out_features=out_features,
            n_layers=self.n_core_layers,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
            init_std=init_std,
        )
        # add the final log conc and log rate readout layers
        # this will make the total number of layers n_layers
        # extract the hidden features from the core module
        self.hidden_features = self.core_module.hidden_features
        self.pre_conc_module = nn.Linear(self.hidden_features, out_features)
        self.pre_rate_module = nn.Linear(self.hidden_features, out_features)
        # initialize the weights
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        core_out = self.core_module(x)
        pre_conc = self.pre_conc_module(core_out)
        pre_rate = self.pre_rate_module(core_out)

        # clamp pre_conc if self.clamp_pre_conc is True
        pre_conc = (
            pre_conc.clamp(max=self.pre_conc_max) if self.clamp_pre_conc else pre_conc
        )

        # clamp pre_rate if self.clamp_pre_rate is True
        pre_rate = (
            pre_rate.clamp(min=self.pre_rate_min) if self.clamp_pre_rate else pre_rate
        )

        # apply the nonzero transforms to get conc and rate
        conc = nonneg_transform_lookup[self.nonneg_transform](pre_conc)
        rate = nonneg_transform_lookup[self.nonneg_transform](pre_rate)

        # up-clamp conc and rate below ensure they're always positive
        finfo = torch.finfo(rate.dtype)
        conc = conc.clamp(min=finfo.tiny)
        rate = rate.clamp(min=finfo.tiny)
        return conc, rate


# suited for monkey data
# where images are square 41x41
# TODO: make this more general?
class MonkeyMeanStdTconv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_dim,
        nonlinearity,
        dropout_rate,
        nonneg_std_transform,
        init_std=1e-3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dim = kernel_dim
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        self.nonneg_std_transform = nonneg_std_transform
        self.init_std = init_std

        self.out_height = int(math.sqrt(out_features))
        self.out_width = self.out_height  # square images

        self.n_layers, remainder = divmod(self.out_height - 1, self.kernel_dim - 1)
        if remainder != 0:
            raise ValueError(
                "Kernel size does not fit stimulus size without padding and/or striding. Try a different kernel size."
            )

        if self.n_layers > 8:
            raise ValueError("Too many layers. Try a smaller kernel size.")

        kernel_height = self.kernel_dim
        kernel_width = self.kernel_dim
        self.kernel_size = (kernel_height, kernel_width)

        self.in_channels = in_features
        self.out_channels = 2**self.n_layers

        self.core_module = nn.Sequential()

        for i in range(self.n_layers):
            self.core_module.add_module(
                f"tconv_{i}",
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                ),
            )
            self.core_module.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            self.core_module.add_module(
                f"nonlinearity_{i}", nonlinearity_lookup[nonlinearity]()
            )
            self.in_channels = self.out_channels
            self.out_channels = self.out_channels // 2

        self.mean_module = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=(1, 1),
        )
        self.pre_std_module = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=(1, 1),
        )

        # TODO: what's the right weight initialization?

    def forward(self, x):
        x_channeled = x.view(-1, self.in_features, 1, 1)
        core_out = self.core_module(x_channeled)
        mean = self.mean_module(core_out).flatten(start_dim=1)
        pre_std = self.pre_std_module(core_out).flatten(start_dim=1)
        # apply the nonzero transform to the pre_std to get std
        std = nonneg_transform_lookup[self.nonneg_std_transform](pre_std)
        # clamp std below ensure it's always positive
        finfo = torch.finfo(std.dtype)
        std = std.clamp(min=finfo.tiny)
