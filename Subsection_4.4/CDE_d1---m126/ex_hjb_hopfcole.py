import torch

from ex_meta import HJB, LinearPDE, diag_curve, manifold_curve


class HjbHopfCole1a(HJB):
    name = 'HjbHopfCole1a'
    musys_depends_on_v = True
    sgmsys_depends_on_v = False
    f_depends_on_v = True
    compute_cost_gap = None

    x0_for_train = {'S2': diag_curve, 'S3': manifold_curve}

    nsamp_mc = 10**6
    num_testpoint = 300

    coeff_sgm = 0.5
    c_val = 2

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        assert self.sgmsys_depends_on_v is False

        self.dim_w = dim_x
        self.dim_u = dim_x

    def b(self, t, x):
        i = torch.arange(self.dim_x, device=x.device)
        return torch.sin(t + i + torch.roll(x, -1, dims=-1))

    def sgm_sys(self, t, x, _k, dw):
        # dist = (x - torch.sin(torch.pi * t)).pow(2).mean(-1, keepdim=True)
        # return self.coeff_sgm * torch.tanh(dist) * dw
        return self.coeff_sgm * dw

    def mu_sys(self, t, x, k):
        btx = self.b(t, x)
        sgm_k = self.sgm_sys(t, x, None, k)
        return btx + self.c_val * sgm_k

    def mu_pil(self, t, x):
        return self.b(t, x)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def g(self, x):
        g1 = 1 + (x**2).mean(-1, keepdim=True)
        g2 = 0.5 * torch.sin(10 * x).mean(-1, keepdim=True)
        return g1 + g2

    def v_term(self, x):
        return torch.log(self.g(x))

    def f(self, _t, _x, k):
        return 0.5 * k.pow(2).sum(-1, keepdims=True)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        class LinPDE(LinearPDE):
            musys_depends_on_v = False
            sgmsys_depends_on_v = False
            f_depends_on_v = False
            nsamp_mc = self.nsamp_mc

            def v_term(lpde_self, x):
                return self.g(x)**(-self.c_val**2)

            def f(lpde_self, t, x, _v):
                return torch.zeros_like(x[..., [0]])

            def mu_pil(lpde_self, _t, _x):
                return lpde_self.mu_sys(_t, _x, None)

            def sgm_pil(lpde_self, _t, _x, dw):
                return lpde_self.sgm_sys(_t, _x, None, dw)

            def mu_sys(lpde_self, t, x, _v):
                return self.b(t, x)

            def sgm_sys(lpde_self, t, x, _v, dw):
                return self.sgm_sys(t, x, None, dw)

            def x0_points(lpde_self, num_points):
                return self.x0_points(num_points)

        lin_pde = LinPDE(self.dim_x,
                         t0=self.t0,
                         te=self.te,
                         use_dist=self.use_dist)
        v_inner = lin_pde.v(t, x)
        v_val = -self.c_val**(-2) * torch.log(v_inner)
        return v_val


class HjbHopfColeSumG(HjbHopfCole1a):
    sphere = False

    def g(self, x):
        i = torch.arange(self.dim_x).float()
        sin_i = torch.sin(i * torch.pi / self.dim_x)
        rad_2 = (sin_i + x).pow(2).sum(-1, keepdim=True) * self.dim_x**(-0.5)
        g1 = 1. + rad_2
        g2 = -0.8 * torch.cos(torch.pi * self.dim_x**(-0.5) * rad_2)
        return g1 + g2


class HjbHopfCole1aBmPil(HjbHopfCole1a):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw


class HjbHopfCole1b(HjbHopfCole1a):
    coeff_sgm = 0.2


class HjbHopfCole1c(HjbHopfCole1a):
    coeff_sgm = 0.025


class HjbHopfCole2(HjbHopfCole1a):

    def b(self, _t, x):
        return torch.sin(torch.roll(x, -1, dims=-1))

    def sgm_sys(self, t, x, _k, dw):
        distance = (t - 0.5)**2 + x.pow(2).mean(-1, keepdim=True)
        return self.coeff_sgm * torch.tanh(distance) * dw


class HjbHopfCole2BmPil(HjbHopfCole2):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw


class HjbHopfCole3(HjbHopfCole1a):

    def sgm_sys(self, t, x, _k, dw):
        distance = (t - 0.5)**2 + x.pow(2).mean(-1, keepdim=True)
        return self.coeff_sgm * torch.tanh(distance) * dw


class HjbHopfCole4a(HjbHopfCole1a):
    x0_for_train = {'S2': diag_curve}

    def b(self, t, x):
        i = torch.arange(self.dim_x, device=x.device)
        return torch.sin(2 * torch.pi * t + i + torch.roll(x, -1, dims=-1))

    def sgm_sys(self, t, x, _k, dw):
        dim_sqrt = self.dim_w**0.5
        x_roll = torch.roll(x, -1, dims=-1)

        distance = (t - 0.5)**2 + x.pow(2).mean(-1, keepdim=True)
        sin_dist = torch.sin(distance)
        sinx = torch.sin(x_roll)
        cosx = torch.cos(x_roll)
        val = cosx * dw.sum(-1, keepdims=True) / dim_sqrt
        val = val + sin_dist / dim_sqrt * (sinx * dw).sum(-1, keepdims=True)
        return self.coeff_sgm * val


class HjbHopfCole4aBmPil(HjbHopfCole4a):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw


class HjbHopfCole4b(HjbHopfCole4a):
    coeff_sgm = 0.025


class HjbHopfCole4bBmPil(HjbHopfCole4b):

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, t, x, dw):
        return dw
