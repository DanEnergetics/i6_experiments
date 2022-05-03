from returnn_common import nn
import numpy as np


def normal_distribution(x: nn.Tensor, *, mu: nn.Tensor, sigma: nn.Tensor) -> nn.Tensor:

    sub = nn.combine(x, mu, kind="sub", allow_broadcast_all_sources=True)
    sub = -0.5 * (sub ** 2.0)
    sigma_s = sigma ** 2.0
    div = nn.combine(sub, sigma_s, kind="truediv", allow_broadcast_all_sources=True)
    return nn.exp(div) / (2.0 * np.pi * sigma_s) ** 0.5  # [B, T, N]


class VarianceNetwork(nn.Module):
    """
    Predicts the variance for the upsampling
    """

    def __init__(self):
        super().__init__()
        self.lstm_dim = nn.FeatureDim("lstm_dim", 512)
        self.lstm_1 = nn.LSTM(out_dim=self.lstm_dim)
        self.lstm_2 = nn.LSTM(out_dim=self.lstm_dim)
        self.linear = nn.Linear(out_dim=nn.FeatureDim("gauss_variance", 1))

    def __call__(self, inp: nn.Tensor, durations: nn.Tensor, time_dim: nn.Dim):
        """

        :param inp: H, [B, N, F]
        :param durations: d, [B, N]
        :return: variance [B, N]
        """
        durations = nn.expand_dim(durations, dim=nn.FeatureDim("dur", 1))
        cat = nn.concat((inp, inp.feature_dim), (durations, durations.feature_dim))

        fw_1, _ = self.lstm_1(cat, axis=time_dim, direction=1)
        bw_1, _ = self.lstm_1(cat, axis=time_dim, direction=-1)
        cat = nn.concat((fw_1, fw_1.feature_dim), (bw_1, bw_1.feature_dim))

        fw_2, _ = self.lstm_2(cat, axis=time_dim, direction=1)
        bw_2, _ = self.lstm_2(cat, axis=time_dim, direction=-1)
        cat = nn.concat((fw_2, fw_2.feature_dim), (bw_2, bw_2.feature_dim))

        lin = self.linear(cat)
        # Softplus activation
        var = nn.log(nn.exp(lin) + 1.0) + 10.0 ** -12
        var = nn.squeeze(var, axis=var.feature_dim)
        return var  # [B, N]


class GaussianUpsampling(nn.Module):
    """
    Performs gaussian upsampling for given input, duration and (learned) variances
    """

    def __init__(self):
        super().__init__()
        self.out_time_axis = nn.SpatialDim("upsamling_time")

    @nn.scoped
    def __call__(
        self,
        inp: nn.Tensor,
        durations: nn.Tensor,
        variances: nn.Tensor,
        time_dim: nn.Dim,
    ) -> nn.Tensor:
        """

        :param inp: H, [B, N, F]
        :param durations: d, [B, N]
        :param time_dim: Time dimension of the input, denoted by N
        :return: upsampled input
        """

        c = nn.cumsum(durations, axis=time_dim) - (0.5 * durations)  # [B,N]

        l = nn.reduce(durations, mode="sum", axis=time_dim, use_time_mask=True)  # [B]
        l = nn.cast(l, dtype="int32")

        t, _ = nn.range_from_length(l, out_spatial_dim=self.out_time_axis)  # [B,T]
        t = nn.cast(t, dtype="float32")

        w_t = normal_distribution(t, mu=c, sigma=variances)

        w_t = w_t / nn.combine(
            nn.math_norm(w_t, p=1, axis=time_dim), 10 ** -16, kind="maximum"
        )
        output = nn.dot(w_t, inp, reduce=time_dim)  # [B, T, F]

        return output
