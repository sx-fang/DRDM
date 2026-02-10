# Some examples of PDE and SOCP
import torch

from ex_meta import PDEwithVtrue, diag_curve, e1_curve


class QLP1(PDEwithVtrue):
    '''
    This example and its derived class are modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'QLP-1'
    musys_depends_on_v = False
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    x0_for_train = {'S2': diag_curve}
    x0pil_range = 1.5

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        # The definition of self.ci differs from the example used in SDGD,
        # as our test typically involves solving v(0, x)
        # for x located on the diagonal of a cube, rather than within a ball.
        self.ci = self.coeff_of_v()

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, self.dim_x)
        ci = (1.5 + torch.sin(xi)) / self.dim_x
        return ci

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return 2**0.5 * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return v * dw

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1

        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        d_v = sumdx_v + 0.5 * v.pow(2) * sumdxx_v
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        return v - v.pow(3) - dv_val - v_true + v_true.pow(3)


class QLP2a(QLP1):
    x0_for_train = {'S2': diag_curve}

    name = 'QLP-2a'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return v * torch.ones_like(x) * dw

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1
        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        d_v = 0.5 * (v * sumdx_v + v.pow(2) * sumdxx_v)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class QLP2b(QLP2a):
    name = 'QLP-2b'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def sgm_sys(self, _t, x, v, dw):
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        val = cosx * dw.mean(-1, keepdims=True)
        val = val + v * (sinx * dw).mean(-1, keepdims=True)
        return val

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1
        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        sgm_1 = (v * sinx).pow(2).mean(-1, keepdims=True)
        sgm_2 = v * sinx.mean(-1, keepdims=True)

        sgm2_ii = cosx.pow(2) + sgm_1 + sgm_2 * (2 * cosx)
        sgm2_ir1 = cosx * cosx_r1 + sgm_1 + sgm_2 * (cosx + cosx_r1)

        dxixi_v = (self.ci * sgm2_ii * sxinn_cxinn).sum(-1, keepdim=True)
        dxir1_v0 = -sin_xinn * dx_xinn1 * dx_xinn2 - cos_xinn * sinx_r1
        dxir1_v = (self.ci * sgm2_ir1 * dxir1_v0).sum(-1, keepdim=True)
        tr_hess = (dxixi_v + 2 * dxir1_v) / self.dim_x

        d_v = 0.5 * (v * sumdx_v + tr_hess)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class QLP1_UnitBall(QLP1):
    x0_for_train = {'S1': e1_curve, 'S2': diag_curve}
    sphere = True

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi, torch.pi, self.dim_x)
        ci = 2 * self.dim_x**(-0.5) * torch.sin(xi)**2
        return ci


class QLP1_varci(QLP1):

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi, torch.pi, self.dim_x)
        ci = (1.5 + torch.sin(xi)) / self.dim_x
        return ci


class QLP2a_varci(QLP2a):

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi, torch.pi, self.dim_x)
        ci = (1.5 + torch.sin(xi)) / self.dim_x
        return ci


class QLP2b_varci(QLP2b):

    def coeff_of_v(self):
        xi = torch.linspace(-torch.pi, torch.pi, self.dim_x)
        ci = (1.5 + torch.sin(xi)) / self.dim_x
        return ci
