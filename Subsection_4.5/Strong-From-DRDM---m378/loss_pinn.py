import torch
from torch.func import grad_and_value, jacrev
from loss_meta import LossCollection
from ex_meta import isin_ddp


def dt_dx_dxx_single(func, t, x):
    # t shape: [1]
    # x shape: [dim_x]
    # func: (t, x) -> y, y shape: []
    # output: (\partial_t, \partial_x, \partial_xx) func(t, x), with shape [], [dim_x], [dim_x, dim_x]

    def wrap_func(_t, _x):
        y = func(_t, _x).squeeze()
        return y, y

    func_dy_y = grad_and_value(wrap_func, argnums=(0, 1), has_aux=True)

    def wrap_dy_y(_t, _x):
        (dydt, dydx), (y, _) = func_dy_y(_t, _x)
        return dydx, (y, dydt, dydx)

    func_ddy_y = jacrev(wrap_dy_y, argnums=1, has_aux=True)
    dydxx, (y, dydt, dydx) = func_ddy_y(t, x)
    return y, dydt, dydx, dydxx


def dt_dx_dxx(func, t, x):
    # t shape: [some batch dims, 1]
    # x shape: [some batch dims, dim_x]
    # func: (t, x) -> y, y shape: [some batch dims, 1]
    # output: y, dydt, dydx, dydxx with shape [some batch dims, 1], [some batch dims, 1, dim_x], [some batch dims, 1, dim_x, dim_x]

    req_grad_t = t.requires_grad
    req_grad_x = x.requires_grad
    t.requires_grad_(True)
    x.requires_grad_(True)

    dim_x = x.shape[-1]
    batch_shape = t.shape[:-1]
    t_flat = t.reshape(-1, 1)
    x_flat = x.reshape(-1, dim_x)

    y, dydt, dydx, dydxx = torch.vmap(
        dt_dx_dxx_single,
        in_dims=(None, 0, 0),
        out_dims=0,
    )(func, t_flat, x_flat)
    y = y.reshape(batch_shape + (1, ))
    dydt = dydt.reshape(batch_shape + (1, ))
    dydx = dydx.reshape(batch_shape + (1, dim_x))
    dydxx = dydxx.reshape(batch_shape + (1, dim_x, dim_x))

    t.requires_grad_(req_grad_t)
    x.requires_grad_(req_grad_x)
    return y, dydt, dydx, dydxx


class QuasiPinn(LossCollection):
    name = 'PINN-for-Quasi-Linear-PDE'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isin_ddp() is False, f"{self} does not support DDP currently."

    def log_func(self):
        return {'pde_loss': self._pde_loss}

    def init_train(self):
        self.vnn = self.net_dict['vnn']
        self.unn = None
        for net in self.net_dict.values():
            net.train()

    def loss_desc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        t_expand = self.t_path.expand_as(xt_pil[..., :1])
        t, x = t_expand[:-1].detach(), xt_pil[:-1].detach()
        t_term = t_expand[-1, ...].detach()
        x_term = xt_pil[-1, ...].detach()

        v, dvdt, dvdx, dvdxx = dt_dx_dxx(self.vnn, t, x)
        drift = torch.einsum(
            '...i, ...ji->...j',
            self.problem.mu_sys(t, x, v),
            dvdx,
        )
        sgm2dydxx = self.problem.tr_sgm2vxx(t, x, v, dvdxx)
        dv = dvdt + drift + 0.5 * sgm2dydxx

        if self.problem.ceoff_vx != 0.:
            dv = dv + self.problem.ceoff_vx * (dvdx.pow(
                self.problem.exponential_vx)).sum(-1)

        f = self.problem.f(t, x, v)
        loss_pde = (dv + f).pow(2).mean()
        if hasattr(self.problem, 'v_term'):
            loss_pde = loss_pde + (self.problem.v_term(x_term) -
                                   self.vnn(t_term, x_term)).pow(2).mean()

        # ========== log ==========
        self._pde_loss = loss_pde.detach().abs().item()
        return loss_pde

    def loss_asc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        return 'no asc loss'


class SocPinn(QuasiPinn):
    name = 'PINN-for-HJB'

    def log_func(self):
        return {
            'pde_loss': self._pde_loss,
            'ctr_loss': self._ctr_loss,
        }

    def init_train(self):
        self.vnn = self.net_dict['vnn']
        self.unn = self.net_dict['unn']
        for net in self.net_dict.values():
            net.train()

    def loss_desc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        t, x = self.t_path[:-1].detach(), xt_pil[:-1].detach()
        t_term = self.t_path[-1, ...].detach()
        x_term = xt_pil[-1, ...].detach()

        t_expand = self.t_path.expand_as(xt_pil[..., :1])
        t, x = t_expand[:-1].detach(), xt_pil[:-1].detach()
        t_term = t_expand[-1, ...].detach()
        x_term = xt_pil[-1, ...].detach()

        u = self.unn(t, x)
        mu = self.problem.mu_sys(t, x, u)
        _, dvdt, dvdx, dvdxx = dt_dx_dxx(self.vnn, t, x)
        drift = torch.einsum('...i, ...ji->...j', mu.detach(), dvdx)
        sgm2dydxx = self.problem.tr_sgm2vxx(t, x, u.detach(), dvdxx)
        dv = dvdt + drift + 0.5 * sgm2dydxx

        if self.problem.ceoff_vx != 0.:
            dv = dv + self.problem.ceoff_vx * (dvdx.pow(
                self.problem.exponential_vx)).sum(-1)

        f = self.problem.f(t, x, u)
        loss_pde = (dv + f.detach()).pow(2).mean()
        loss_pde = loss_pde + (self.problem.v_term(x_term) -
                               self.vnn(t_term, x_term)).pow(2).mean()

        cost = torch.einsum('...i, ...ji->...j', mu, dvdx.detach())
        cost = cost + 0.5 * self.problem.tr_sgm2vxx(t, x, u, dvdxx.detach())
        ctr_loss = (cost + f).mean()
        loss_tot = loss_pde + ctr_loss
        # ========== log ==========
        self._pde_loss = loss_pde.detach().abs().item()
        self._ctr_loss = ctr_loss.detach().item()

        return loss_tot
