import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from neuralode.utils import euler, runge_kutta, _check_none_zero


class OdeWithGrad(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        batch_size = z.shape[0]
        out = self.forward(z, t)

        a = grad_outputs
        a_df_dz, a_df_dt, *a_df_dp = torch.autograd.grad((out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True)

        if a_df_dp is not None:
            a_df_dp = torch.cat([p_grad.flatten() for p_grad in a_df_dp]).unsqueeze(0)
            a_df_dp = a_df_dp.expand(batch_size, -1) / batch_size
        if a_df_dt is not None:
            a_df_dt = a_df_dt.expand(batch_size, 1) / batch_size
        return out, a_df_dz, a_df_dt, a_df_dp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)


class OdeAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, t, flat_parameters, func, tol, solver):
        assert isinstance(func, OdeWithGrad)
        bs, *z_shape = y0.size()
        time_len = t.size(0)

        with torch.no_grad():
            y = torch.zeros(time_len, bs, *z_shape).to(y0)
            y[0] = y0
            for i_t in range(time_len - 1):
                y0 = solver(y0, t[i_t], t[i_t + 1], func, tol)
                y[i_t + 1] = y0

        ctx.func = func
        ctx.solver = solver
        ctx.save_for_backward(t, y.clone(), flat_parameters)
        ctx.tol = tol
        return y

    @staticmethod
    def backward(ctx, grad_output):
        func = ctx.func
        solver = ctx.solver
        t, y, flat_parameters = ctx.saved_tensors
        time_len, batch_size, *y_shape = y.size()
        n_dim = np.prod(y_shape)
        n_params = flat_parameters.size(0)

        def augmented_dynamics(aug_y_i, t_i):
            y_i, a = aug_y_i[:, :n_dim], aug_y_i[:, n_dim:2 * n_dim]

            y_i = y_i.view(batch_size, *y_shape)
            a = a.view(batch_size, *y_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                y_i = y_i.detach().requires_grad_(True)
                func_eval, df_dy, df_dt, df_dp = func.forward_with_grad(y_i, t_i, grad_outputs=a)
                df_dy = _check_none_zero(df_dy, (batch_size, *y_shape), y_i)
                df_dt = _check_none_zero(df_dt, (batch_size, 1), y_i)
                df_dp = _check_none_zero(df_dp, (batch_size, n_params), y_i)

            func_eval = func_eval.view(batch_size, n_dim)
            df_dy = df_dy.view(batch_size, n_dim)
            return torch.cat((func_eval, -df_dy, -df_dp, -df_dt), dim=1)

        dL_dy = grad_output.view(time_len, batch_size, n_dim)
        with torch.no_grad():
            adj_y = torch.zeros(batch_size, n_dim).to(dL_dy)
            adj_p = torch.zeros(batch_size, n_params).to(dL_dy)
            adj_t = torch.zeros(time_len, batch_size, 1).to(dL_dy)

            for i_t in range(time_len - 1, 0, -1):
                y_i = y[i_t]
                t_i = t[i_t]
                f_i = func(y_i, t_i).view(batch_size, n_dim)
                dLdy_i = dL_dy[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdy_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # use  adjoints with gradients dL/di and dynamics
                adj_y += dLdy_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i
                aug_y = torch.cat((y_i.view(batch_size, n_dim), adj_y,
                                   torch.zeros(batch_size, n_params).to(y), adj_t[i_t]), dim=-1)
                aug_ans = solver(aug_y, t_i, t[i_t - 1], augmented_dynamics, eps=ctx.tol)

                adj_y[:] = aug_ans[:, n_dim:2 * n_dim]
                adj_p[:] += aug_ans[:, 2 * n_dim:2 * n_dim + n_params]
                adj_t[i_t - 1] = aug_ans[:, 2 * n_dim + n_params:]

                del aug_y, aug_ans
            dLdy_0 = dL_dy[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdy_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            adj_y += dLdy_0
            adj_t[0] = adj_t[0] - dLdt_0

        return adj_y.view(batch_size, *y_shape), adj_t, adj_p, None, None, None


class NeuralODE(nn.Module):
    def __init__(self, func, tol=1e-3, solver='euler'):
        super(NeuralODE, self).__init__()
        assert isinstance(func, OdeWithGrad)
        self.func = func
        self.tol = tol
        self.solver = runge_kutta if solver == 'runge_kutta' else euler

    def forward(self, z0, t=torch.Tensor([0., 1.]), sequence=False):
        t = t.to(z0)
        z = OdeAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.tol, self.solver)
        if sequence:
            return z
        else:
            return z[-1]