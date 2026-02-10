import torch

from ex_meta import LinearPDE, PDEwithVtrue, diag_curve, e1_curve


class Shock(LinearPDE):
    name = 'Shock'
    x0_for_train = {'S2': diag_curve}
    record_linf_error = False

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False
    delta = 0.1

    coeff_mu = 1.
    num_testpoint = 300
    nsamp_mc = 10**6

    # ylim_xt_scatter = (-4.1, 4.1)

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        self.te = 2

    def mu_pil(self, t, x):
        return self.mu_sys(t, x, None)

    def sgm_pil(self, _t, _x, dw):
        return self.delta**(0.5) * dw

    def mu_sys(self, _t, x, _v):
        mu = self.coeff_mu * torch.tanh(10 * x)
        return mu

    def sgm_sys(self, _t, _x, _v, dw):
        return self.delta**(0.5) * dw

    def v_term(self, x):
        tanhx = torch.tanh(x).mean(-1, keepdim=True)
        osc = torch.cos(10 * x).mean(-1, keepdim=True)
        return osc + tanhx

    def f(self, _t, x, _v):
        return torch.zeros_like(x[..., [0]])


class Shock1a(Shock):
    pass


class Shock1aBmPil(Shock1a):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw


class Shock1b(Shock):
    coeff_mu = 5


class Shock1bBmPil(Shock1b):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, _x, dw):
        return dw


class Counter(PDEwithVtrue):
    name = 'Counter_Example'
    x0_for_train = {'S1': e1_curve, 'S2': diag_curve}

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.ones_like(x) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, _v, dw):
        return torch.ones_like(x) * dw

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        dt_u = torch.cos(t + x).mean(-1, keepdim=True)
        dxx_u = -torch.sin(t + x).mean(-1, keepdim=True)
        return -dt_u - 0.5 * dxx_u

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class LinearSin(PDEwithVtrue):
    name = 'Linear_Sinx'
    x0_for_train = {'diag': diag_curve}

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False
    c_in_sgm = 5.

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        mu = torch.sin(2 * x)
        return mu

    def sgm_pil(self, t, x, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def mu_sys(self, _t, x, _v):
        mu = torch.sin(2 * x)
        return mu

    def sgm_sys(self, t, x, _v, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)

        mu = torch.sin(2 * x)
        mu_vx = (mu * costx).mean(-1, keepdim=True)

        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)
        return -dt_v - mu_vx - 0.5 * tr_sgm_vxx

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)
