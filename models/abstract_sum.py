import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import softplus
from torch.distributions.uniform import Uniform


def sum(x, dim=-1):
    # if torch.isnan(x).any(): raise ValueError
    result = x.sum(dim)
    mask1 = x == float('inf')
    mask2 = x == float('-inf')
    inf_mask = torch.logical_or(mask1, mask2).any(dim)
    if inf_mask.any(): print("inf detected!")
    result[inf_mask] = ((mask1.sum(dim) - mask2.sum(dim)) * float('inf'))[inf_mask]
    result[torch.isnan(result)] = 0
    return result


class AbstractSum(torch.nn.Module):
    def __init__(self, in_channels, num_basis, basis='powerlaw', mc_num_surv=1, mc_num_time=3):
        super(AbstractSum, self).__init__()

        self.num_basis = num_basis

        if basis == 'exp':
            self.num_param = 2
        elif basis == 'sigmoid':
            self.num_param = 3
        elif basis == 'cosine' or basis == 'cos':
            self.num_param = 3
        elif basis == 'relu':
            self.num_param = 3
        elif basis == 'powerlaw':
            self.num_param = 2
        elif basis == 'mixed':
            self.num_param = 5
        else:
            raise NotImplementedError

        self.basis = basis

        self.weight_layer = Linear(in_channels, self.num_basis * self.num_param)
        # torch.nn.init.zeros_(self.weight_layer.weight)
        # torch.nn.init.zeros_(self.weight_layer.bias)
        self.psai = torch.nn.Parameter(torch.tensor(1).float())
        self.mc_num_surv = mc_num_surv
        self.mc_num_time = mc_num_time
        # self.lin_dst = Linear(in_channels, in_channels)
        # self.lin_final = Linear(in_channels, 1)

    def kernel(self, tau, weights):

        if self.basis == 'exp':
            alphas, betas = weights
            out = alphas * torch.exp(betas * tau)
            # print(f'def kernel: out:{out}, weights:{weights}')
            # if torch.isnan(out).any() or torch.isinf(out).any(): print(f'def kernel: out:{out}')
        elif self.basis == 'sigmoid':
            alphas, betas, delta = weights
            out = alphas * torch.sigmoid(betas * tau + delta)
        elif self.basis == 'cosine' or self.basis == 'cos':
            alphas, betas, delta = weights
            out = alphas * torch.cos(betas * tau + delta)
        elif self.basis == 'relu':
            alphas, betas, gamma = weights
            out = gamma * torch.relu(alphas * tau + betas)
        elif self.basis == 'powerlaw':
            alphas, betas = weights
            out = alphas * (1 + tau) ** (- betas ** 2)
        elif self.basis == 'mixed':
            alphas, betas, gamma, delta, eta = weights
            out = gamma * torch.relu(alphas * tau + betas) + delta * (1 + tau) ** (- eta ** 2)
        else:
            raise NotImplementedError

        # out = torch.clamp(out, min=-1e+2, max=1e+2)
        #
        # out_abs = torch.abs(out)
        # out_abs = torch.clamp(out_abs, min=1, max=100)
        # out = torch.sign(out) * out_abs

        # print(f'out: {out}')
        # out[torch.logical_or(torch.isnan(out), torch.isinf(out))] = 0
        if torch.isnan(out).any() or torch.isinf(out).any(): print("!!!! Stilled bad values exist")
        return out

    def to_positive(self, x):
        s = 1 / 1.5
        return torch.nn.functional.softplus(x, beta=s)

    def forward(self, z_src, z_dst, delta_t):

        return self.intensity_func(delta_t, z_src, z_dst, return_params=False)

    def intensity_func(self, delta_t, z_src=None, z_dst=None, params=None, return_params=True):
        if z_src is None or z_dst is None:
            assert params is not None
            # if torch.isnan(params[0]).any(): print(f"def initensity_func: input params:{params}")
        if params is None:
            # if torch.isnan(z_src).any() or torch.isnan(z_dst).any(): print(
            #     f"def intensity_func: input z_src: {z_src}, input z_dst: {z_dst}")
            weights = self.weight_layer(torch.cat((z_src, z_dst), dim=-1))  # (batch_size, num_param*num_basis)
            params = torch.split(weights, self.num_basis,
                                 dim=-1)  # num_param-sized tuple of tensor(batch_size, num_basis)
            # if torch.isnan(params[0]).any(): print(
            #     f"def intensity_func: computed params: {params}")
        basis_vals = self.kernel(delta_t.unsqueeze(-1), params)  # (*, num_basis)
        # int_vals = self.to_positive(basis_vals.sum(-1))
        int_vals = self.to_positive(sum(basis_vals))
        # if torch.isnan(int_vals).any():
        #     # print(f"def intensity_func: basis_vals: {basis_vals}, delta_t: {delta_t}, params:{params}")
        #     print(f"def intensity_func: basis_vals: {basis_vals}")
        #     print(f"def intensity_func: basis_vals.sum(-1): {basis_vals.sum(-1)}, int_vals: {int_vals}")

        if return_params:
            return int_vals, params
        else:
            return int_vals

    def intensity_func_dyrep(self, delta_t, z_src=None, z_dst=None, params=None, return_params=True):
        if z_src is None or z_dst is None:
            assert params is not None
        if params is None:
            params = self.weight_layer(torch.cat((z_src, z_dst), dim=-1)).squeeze(
                -1)  # (batch_size, num_param*num_basis)
        int_vals = softplus(params, beta=self.psai)
        if return_params:
            return int_vals, params
        else:
            return int_vals

    def compensator_func(self, delta_t, z_src=None, z_dst=None, params=None):

        if z_src is None or z_dst is None:
            assert params is not None
            # if torch.isnan(params[0]).any(): print(f"def compensator_func: input params:{params}")
        if params is None:
            # if torch.isnan(z_src).any() or torch.isnan(z_dst).any(): print(
            #     f"def compensator_func: input z_src: {z_src}, input z_dst: {z_dst}")
            weights = self.weight_layer(torch.cat((z_src, z_dst), dim=-1))  # (batch_size, num_param*num_basis)
            params = torch.split(weights, self.num_basis,
                                 dim=-1)  # num_param-sized tuple of tensor(batch_size, num_basis)
            # if torch.isnan(params[0]).any(): print(
            #     f"def compensator_func: computed params: {params}")
        mc_delta_t = delta_t.unsqueeze(-1).repeat_interleave(self.mc_num_surv, dim=-1)  # (batch_size, mc_num)

        sampler = Uniform(torch.zeros_like(mc_delta_t).float(), mc_delta_t.float())
        mc_points = sampler.sample()

        # num_param-sized list of tensor(batch_size, mc_num, num_basis)
        mc_params = [p.unsqueeze(-2).repeat_interleave(self.mc_num_surv, dim=-2) for p in params]

        # if torch.isnan(mc_params[0]).any(): print(
        #     f"def compensator_func: mc_params: {mc_params}")

        mc_int = self.intensity_func(mc_points, params=mc_params, return_params=False)

        compensator = mc_int.sum(-1) * delta_t / self.mc_num_surv

        # if torch.isnan(compensator).any():
        #     print(f"def compensator_func: mc_points:{mc_points} mc_int: {mc_int}, "
        #           f"delta_t: {delta_t}, self.mc_num_surv: {self.mc_num_surv}")

        return compensator

    def ll(self, z_src, z_dst, delta_t):

        # if torch.isnan(z_src).any() or torch.isnan(z_dst).any():
        #     print(f"def ll: input z_src: {z_src}, input z_dst: {z_dst}")

        int_vals, params = self.intensity_func(delta_t, z_src=z_src, z_dst=z_dst, return_params=True)

        # if torch.tensor([torch.isnan(params[i]).any() for i in range(len(params))
        #                  ]).any(): print(f"def ll: computed params: {params}")

        # print(f"def ll: int_vals: {int_vals}")
        int_log = torch.log(int_vals + 1e-8)
        compensator = self.compensator_func(delta_t, params=params)

        return int_log - compensator

    def likelihood_func(self, delta_t, z_src=None, z_dst=None, params=None):

        int_vals, params = self.intensity_func(delta_t, z_src, z_dst, params, return_params=True)
        compensator = self.compensator_func(delta_t, params=params)

        out = int_vals * torch.exp(-compensator)

        # if torch.isnan(out).any():
        #     print(f"def likelihood_func: params: {params} int_vals: {int_vals}, compensator: {compensator}")

        return out

    def estimate_time(self, z_src, z_dst):
        weights = self.weight_layer(torch.cat((z_src, z_dst), dim=-1))
        params = torch.split(weights, self.num_basis, dim=-1)

        # if torch.isnan(params[0]).any():
        #     print(f"def estimate_time: computed params:{params}, weights:{weights},"
        #           f" input z_src:{z_src}, input z_dst: {z_dst}")

        assert z_src.shape == z_dst.shape

        # mc_time_delta_t = z_src.unsqueeze(-1).repeat_interleave(self.mc_num_time, dim=-1)  # (batch_size, mc_num)
        # mc_time_s = torch.exp(-mc_time_delta_t) # replace variable t with variable s to compute the improper integral

        sampler = Uniform(torch.zeros(z_src.size(0), self.mc_num_time), torch.ones(z_src.size(0), self.mc_num_time))
        mc_time_points = sampler.sample()
        mc_time_t = - torch.log(mc_time_points)

        # num_param-sized list of tensor(batch_size, mc_num, num_basis)
        mc_time_params = [p.unsqueeze(-2).repeat_interleave(self.mc_num_time, dim=-2) for p in params]

        mc_time_likelihood = self.likelihood_func(mc_time_t, params=mc_time_params)  # (batch_size, mc_num)
        mc_time_integrand = mc_time_t * mc_time_likelihood / mc_time_points

        mc_time_integral = mc_time_integrand.sum(-1) / self.mc_num_time

        # if torch.isnan(mc_time_integral).any():
        #     print(f"def estimate_time:  {mc_time_integrand}, {mc_time_t}, "
        #           f"mc_time_likelihood: {mc_time_likelihood}, {mc_time_points}")

        return mc_time_integral


class DyrepAbstractSum(torch.nn.Module):
    def __init__(self, in_channels, mc_num_surv=1, mc_num_time=3):
        super(DyrepAbstractSum, self).__init__()

        # self.weight_layer = Linear(in_channels, 1, bias=False)
        # self.weight_layer_src = Linear(in_channels // 2, 1, bias=False)
        # self.weight_layer_dst = Linear(in_channels // 2, 1, bias=False)
        # self.psai = torch.nn.Parameter(torch.tensor(1).float())
        self.mc_num_surv = mc_num_surv
        self.mc_num_time = mc_num_time
        # # self.lin_dst = Linear(in_channels, in_channels)
        # # self.lin_final = Linear(in_channels, 1)

        self.lin_src = Linear(in_channels // 2, in_channels // 2)
        self.lin_dst = Linear(in_channels // 2, in_channels // 2)
        self.lin_final = Linear(in_channels // 2, 1)

    def forward(self, z_src, z_dst, delta_t):

        return self.intensity_func(delta_t, z_src, z_dst, return_params=False)

    def intensity_func(self, delta_t, z_src=None, z_dst=None, params=None, return_params=True):

        if z_src is None or z_dst is None:
            assert params is not None
            # if torch.isnan(params[0]).any(): print(f"def initensity_func: input params:{params}")
        if params is None:
            # if torch.isnan(z_src).any() or torch.isnan(z_dst).any(): print(
            #     f"def intensity_func: input z_src: {z_src}, input z_dst: {z_dst}")
            # params = self.weight_layer(torch.cat((z_src, z_dst), dim=-1)).squeeze(-1)
            # # (batch_size, num_param*num_basis)
            # params = self.weight_layer_src(z_src).squeeze(-1) + self.weight_layer_dst(z_dst).squeeze(-1)
            h = self.lin_src(z_src) + self.lin_dst(z_dst)
            h = h.relu()
            params = self.lin_final(h).squeeze(-1)

            # if torch.isnan(params[0]).any(): print(
            #     f"def intensity_func: computed params: {params}")
        # int_vals = self.to_positive(basis_vals.sum(-1))
        # int_vals = self.psai * softplus(params / self.psai)
        # int_vals = params.sigmoid()
        int_vals = softplus(params)
        # if torch.isnan(int_vals).any():
        #     # print(f"def intensity_func: basis_vals: {basis_vals}, delta_t: {delta_t}, params:{params}")
        #     print(f"def intensity_func: basis_vals: {basis_vals}")
        #     print(f"def intensity_func: basis_vals.sum(-1): {basis_vals.sum(-1)}, int_vals: {int_vals}")

        if return_params:
            return int_vals, params
        else:
            return int_vals

    def compensator_func(self, delta_t, z_src=None, z_dst=None, params=None):

        if z_src is None or z_dst is None:
            assert params is not None
            # if torch.isnan(params[0]).any(): print(f"def compensator_func: input params:{params}")
        if params is None:
            # if torch.isnan(z_src).any() or torch.isnan(z_dst).any(): print(
            #     f"def compensator_func: input z_src: {z_src}, input z_dst: {z_dst}")
            # params = self.weight_layer(torch.cat((z_src, z_dst), dim=-1)).squeeze(-1)
            # # (batch_size, num_param*num_basis)
            # params = self.weight_layer_src(z_src).squeeze(-1) + self.weight_layer_dst(z_dst).squeeze(-1)
            h = self.lin_src(z_src) + self.lin_dst(z_dst)
            h = h.relu()
            params = self.lin_final(h).squeeze(-1)
            # if torch.isnan(params[0]).any(): print(
            #     f"def compensator_func: computed params: {params}")
        mc_delta_t = delta_t.unsqueeze(-1).repeat_interleave(self.mc_num_surv, dim=-1)  # (batch_size, mc_num)

        sampler = Uniform(torch.zeros_like(mc_delta_t).float(), mc_delta_t.float())
        mc_points = sampler.sample()

        # num_param-sized list of tensor(batch_size, mc_num, num_basis)
        mc_params = params.unsqueeze(-1).repeat_interleave(self.mc_num_surv, dim=-1)

        # if torch.isnan(mc_params[0]).any(): print(
        #     f"def compensator_func: mc_params: {mc_params}")

        mc_int = self.intensity_func(mc_points, params=mc_params, return_params=False)

        compensator = mc_int.sum(-1) * delta_t / self.mc_num_surv

        # if torch.isnan(compensator).any():
        #     print(f"def compensator_func: mc_points:{mc_points} mc_int: {mc_int},"
        #           f" delta_t: {delta_t}, self.mc_num_surv: {self.mc_num_surv}")

        return compensator

    def ll(self, z_src, z_dst, delta_t):

        # if torch.isnan(z_src).any() or torch.isnan(z_dst).any():
        #     print(f"def ll: input z_src: {z_src}, input z_dst: {z_dst}")

        int_vals, params = self.intensity_func(delta_t, z_src=z_src, z_dst=z_dst, return_params=True)

        # if torch.tensor([torch.isnan(params[i]).any() for i in range(len(params))
        #                  ]).any(): print(f"def ll: computed params: {params}")

        # print(f"def ll: int_vals: {int_vals}")
        int_log = torch.log(int_vals + 1e-8)
        compensator = self.compensator_func(delta_t, params=params)

        return int_log - compensator

    def likelihood_func(self, delta_t, z_src=None, z_dst=None, params=None):

        int_vals, params = self.intensity_func(delta_t, z_src, z_dst, params, return_params=True)
        compensator = self.compensator_func(delta_t, params=params)

        out = int_vals * torch.exp(-compensator)

        # if torch.isnan(out).any():
        #     print(f"def likelihood_func: params: {params} int_vals: {int_vals}, compensator: {compensator}")

        return out

    def estimate_time(self, z_src, z_dst):
        # params = self.weight_layer(torch.cat((z_src, z_dst), dim=-1)).squeeze(-1)
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        params = self.lin_final(h).squeeze(-1)

        # if torch.isnan(params[0]).any():
        #     print(f"def estimate_time: computed params:{params}, weights:{weights},"
        #           f" input z_src:{z_src}, input z_dst: {z_dst}")

        assert z_src.shape == z_dst.shape

        # mc_time_delta_t = z_src.unsqueeze(-1).repeat_interleave(self.mc_num_time, dim=-1)  # (batch_size, mc_num)
        # mc_time_s = torch.exp(-mc_time_delta_t) # replace variable t with variable s to compute the improper integral

        sampler = Uniform(torch.zeros(z_src.size(0), self.mc_num_time), torch.ones(z_src.size(0), self.mc_num_time))
        mc_time_points = sampler.sample()
        mc_time_t = - torch.log(mc_time_points)

        # num_param-sized list of tensor(batch_size, mc_num, num_basis)
        mc_time_params = params.unsqueeze(-1).repeat_interleave(self.mc_num_time, dim=-1)

        mc_time_likelihood = self.likelihood_func(mc_time_t, params=mc_time_params)  # (batch_size, mc_num)
        mc_time_integrand = mc_time_t * mc_time_likelihood / mc_time_points
        # mc_time_integrand = mc_time_likelihood / mc_time_points
        # print(f"mc_time_t:{mc_time_t}, mc_time_likelihood:{mc_time_likelihood}, mc_time_points:{mc_time_points}")
        print(f"params:{params}, intensity:{softplus(params)}")
        mc_time_integral = mc_time_integrand.sum(-1) / self.mc_num_time

        # if torch.isnan(mc_time_integral).any():
        #     print(f"def estimate_time:  {mc_time_integrand}, {mc_time_t}, mc_time_likelihood: "
        #           f"{mc_time_likelihood}, {mc_time_points}")

        return mc_time_integral
