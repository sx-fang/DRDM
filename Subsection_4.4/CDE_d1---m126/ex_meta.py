# Define metaclass for PDEs and related functions
import abc
import gc
from abc import abstractmethod
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.distributed as dist

# plt.rcParams.update({
#     'figure.figsize': (3.5, 3),
#     'font.size': 10,
#     'axes.titlesize': 10,
#     'axes.labelsize': 9,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'legend.fontsize': 8,
#     'savefig.dpi': 600,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.05
# })


def tensor2ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def free_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'xpu':
        torch.xpu.empty_cache()
    elif device.type == 'cpu':
        gc.collect()
    else:
        raise ValueError('The device type is not supported.')


def get_safe_chunksize(num_item_persize: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       use_percent=0.5):
    # num_item_persize: how many items (in dtype) per chunksize

    if device.type == 'cuda':
        avail_mem = torch.cuda.mem_get_info(device)[0]
    elif device.type == 'xpu':
        avail_mem = torch.xpu.mem_get_info(device)[0]
    elif device.type == 'cpu':
        avail_mem = psutil.virtual_memory().available
    else:
        raise ValueError('The device type is not supported.')

    safe_chunksize = avail_mem * use_percent // (num_item_persize *
                                                 dtype.itemsize)
    safe_chunksize = int(safe_chunksize)
    if safe_chunksize == 0:
        raise MemoryError(
            f'The memory of {device} is not enough to get safe_chunksize >= 1.'
        )
    return safe_chunksize


def split_number(num, num_parts):
    part_size = num // num_parts
    parts = [part_size] * num_parts
    for i in range(num % num_parts):
        parts[i] += 1
    return parts


def get_interval(s_range: Union[float, Sequence[float]]):
    if isinstance(s_range, Sequence) is True:
        left_end, right_end = s_range
    else:
        left_end = -s_range
        right_end = s_range
    return left_end, right_end


def e1_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
    sphere=False,
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_1stcoord = torch.zeros([dim_x])
    e_1stcoord[0] = 1.
    if sphere is True:
        e_1stcoord = e_1stcoord / e_1stcoord.norm(dim=-1, keepdim=True)
    xe1 = torch.outer(s_coord, e_1stcoord)
    return s_coord, xe1


def diag_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
    sphere=False,
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_diag = torch.ones((dim_x, ))
    if sphere is True:
        e_diag = e_diag / e_diag.norm(dim=-1, keepdim=True)

    s_coord = torch.linspace(left_end, right_end, num_points)
    xdiag = torch.outer(s_coord, e_diag)
    return s_coord, xdiag


def manifold_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
    sphere=False,
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_vec = torch.arange(1, dim_x + 1)

    x_diag = torch.outer(s_coord, torch.sign(torch.sin(e_vec)))
    x = x_diag + torch.cos(e_vec + s_coord.unsqueeze(-1) * torch.pi)
    if sphere is True:
        x = x / dim_x**0.5
    return s_coord, x


def origin_point(dim_x, num_points):
    s_coord = torch.zeros([num_points])
    x = torch.zeros([num_points, dim_x])
    return s_coord, x


def plot_on_curve(s, v_true, v_approx, sav_prefix, labels=None, ylim=None):
    if labels is None:
        labels = ['True', 'Predicted']
    plt.plot(s, v_true, label=labels[0], color='blue')
    plt.scatter(s, v_approx, label=labels[1], color='red')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(sav_prefix)
    plt.close()


def plot_2d(x,
            y,
            z,
            sav_prefix,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            ylim=None):
    plt.figure()
    plt.scatter(x, y, c=z, cmap='viridis')
    plt.colorbar(label=zlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'{sav_prefix}_2dplot.pdf')
    plt.close()


def modify_ylim(y_min, y_max, rate_edge=0.2, include_zero=True, min_high=1.):
    if y_max - y_min < min_high:
        shift = min_high - (y_max - y_min)
        y_min = y_min - shift / 2
        y_max = y_max + shift / 2

    edge = rate_edge * (y_max - y_min)
    y_min = y_min - edge
    y_max = y_max + edge

    if include_zero is True:
        y_min = min(0, y_min)
        y_max = max(0, y_max)
    return y_min, y_max


class PDE(metaclass=abc.ABCMeta):

    name = 'Default_PDE_Name'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    v_shellfunc = None

    def __init__(self, dim_x, t0=0., te=1., use_dist=False) -> None:
        self.dim_x = dim_x
        self.dim_w = dim_x

        if isinstance(t0, float) or isinstance(t0, int):
            t0 = torch.tensor(t0)
        if isinstance(te, float) or isinstance(te, int):
            te = torch.tensor(te)
        self.t0 = t0
        self.te = te

        self.use_dist = use_dist
        if use_dist is True:
            self.world_size = dist.get_world_size()
            self.save_results = (dist.get_rank() == 0)
        else:
            self.world_size = 1
            self.save_results = True

    @abstractmethod
    def mu_pil(self, _t, _x):
        pass

    @abstractmethod
    def sgm_pil(self, _t, _x, _dw):
        pass

    @abstractmethod
    def mu_sys(self, _t, _x, _k):
        pass

    @abstractmethod
    def sgm_sys(self, _t, _x, _k, _dw):
        pass

    @abstractmethod
    def f(self, _t, _x, _v):
        pass

    @abstractmethod
    def x0_points(self, num_points):
        pass

    def produce_logfunc(self, _v_approx):
        pass

    def produce_results(self, _v_approx, _sav_prefix, ctrfunc_syspath=None):
        pass

    def gen_pilpath(self, x0: torch.Tensor, num_dt: int):
        dt = (self.te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            mu_tn = self.mu_pil(tn, xtn)
            sgm_dwtn = self.sgm_pil(tn, xtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
        t = torch.stack(t, dim=0)
        xt = torch.stack(xt, dim=0)
        # xt: [time, path, dim of x]
        return t, xt

    def gen_syspath(self, x0: torch.Tensor, num_dt: int, v_func):
        dt = (self.te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        vt = [v_func(t[0], x0)]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            vtn = vt[n]
            mu_tn = self.mu_sys(tn, xtn, vtn)
            sgm_dwtn = self.sgm_sys(tn, xtn, vtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
            vt.append(v_func(t[-1], xt[-1]))
        t = torch.stack(t, dim=0)
        xt = torch.stack(xt, dim=0)
        vt = torch.stack(vt, dim=0)
        # xt: [time, path, dim of x]
        return t, xt, vt

    def plot_xtpil(self,
                   sav_prefix,
                   num_path=100,
                   num_dt=16,
                   dim_of_x=-1,
                   ylim=None):
        x0 = self.x0_points(num_path)
        t, xt_pil = self.gen_pilpath(x0, num_dt)

        plt.figure(figsize=(8, 3))
        y_values = xt_pil[..., dim_of_x].cpu().detach().numpy()
        t_values = t.squeeze().cpu().detach().numpy()

        plt.plot(t_values, y_values, 'bo', markerfacecolor='none')
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel('$t$')
        plt.ylabel('$X_t$')
        plt.title('Paths of $X_t$')
        if self.save_results is True:
            plt.savefig(f"{sav_prefix}path_scatter.pdf", dpi=300)
        plt.close()


class PDEwithVtrue(PDE):
    name = 'Default_TVP_Name'

    x0pil_range = 1.
    sphere = False
    x0_for_train = {'S2': diag_curve}

    record_linf_error = True
    num_testpoint = 1000

    @abstractmethod
    def v(self, _t, _x):
        pass

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range, self.sphere)[1]
            for nump, c_func in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(self.num_testpoint)
        vtrue_ontest = self.v(t0, x_test)
        vtrue_ontest_l1 = torch.abs(vtrue_ontest).mean()
        vtrue_ontest_linf = torch.abs(vtrue_ontest).max()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            err = v_approx(t0, x_test) - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            log = {'rel_l1err': l1err.item()}
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func

    def xcurve_for_res(self, num_points=50, x0_range=None):
        if x0_range is None:
            x0_range = self.x0pil_range
        args = (self.dim_x, num_points, x0_range, self.sphere)
        curve_name = list(self.x0_for_train.keys())
        s, xs = zip(*[x0f(*args) for x0f in self.x0_for_train.values()])
        return curve_name, s, xs

    def save_res_oncurve(self, sav_prefix, curve_name, s, vtrue_x0s,
                         vapprox_x0s):
        res_header = []
        res_col = []
        for (cname, si, vtrue, vappr) in \
                zip(curve_name, s, vtrue_x0s, vapprox_x0s):
            res_header.extend(
                [f'coord_{cname}', f'vtrue_{cname}', f'vappr_{cname}'])
            res_col.extend(
                [tensor2ndarray(si),
                 vtrue.squeeze(-1),
                 vappr.squeeze(-1)])
        res_arr = np.stack(res_col, axis=1)
        if self.save_results is True:
            np.savetxt(sav_prefix + 'res_on_line.csv',
                       res_arr,
                       delimiter=',',
                       header=','.join(res_header))

    def res_on_curve(self,
                     s,
                     x_s,
                     curve_name,
                     sav_prefix,
                     v_approx,
                     y_min=None,
                     y_max=None,
                     t=None):
        assert len(s) == len(x_s)
        assert len(x_s) == len(curve_name)

        if t is None:
            t = self.t0
        if type(t) is float:
            t_uns = torch.tensor(t).unsqueeze(-1)
        else:
            t_uns = t.unsqueeze(-1)
        vtrue_xs = [tensor2ndarray(self.v(t_uns, x)) for x in x_s]
        vapprox_xs = [tensor2ndarray(v_approx(t_uns, x)) for x in x_s]

        if (y_min is None) or (y_max is None):
            v_max = np.nanmax([vtrue_xs, vapprox_xs])
            v_min = np.nanmin([vtrue_xs, vapprox_xs])
            v_min, v_max = modify_ylim(v_min, v_max)
            y_min = v_min if y_min is None else y_min
            y_max = v_max if y_max is None else y_max

        if self.save_results is True:
            for i in range(len(s)):
                plot_on_curve(s[i].cpu().numpy(),
                              vtrue_xs[i],
                              vapprox_xs[i],
                              sav_prefix + curve_name[i] + '.pdf',
                              ylim=(y_min, y_max))
            self.save_res_oncurve(sav_prefix, curve_name, s, vtrue_xs,
                                  vapprox_xs)

    def res_on_path(
        self,
        v_approx,
        ctrfunc_syspath=None,
        num_dt=50,
        num_path=8,
    ):
        x0 = self.x0_points(num_path)
        with torch.no_grad():
            if ctrfunc_syspath is None:
                t, xt = self.gen_pilpath(x0, num_dt)
            else:
                t, xt, _ = self.gen_syspath(x0, num_dt, ctrfunc_syspath)
            vt_appr = v_approx(t, xt)
            vt_true = self.v(t, xt)
        err = (vt_appr - vt_true).abs()
        re1_tensor = err.mean(1, keepdims=True) / vt_true.abs().mean(
            1, keepdims=True)

        if self.record_linf_error:
            vture_max = vt_true.abs().max(1, keepdims=True)[0]
            reinf_tensor = err.max(1, keepdims=True)[0] / vture_max

        path_res = {
            't_path': tensor2ndarray(t),
            'vtrue_path': tensor2ndarray(vt_true),
            'vappr_path': tensor2ndarray(vt_appr),
            'rel_l1err_path': tensor2ndarray(re1_tensor),
        }
        if self.record_linf_error:
            path_res['rel_linferr_path'] = tensor2ndarray(reinf_tensor)
        return path_res

    def plot_vtx_1d(self, t_scatter, x_scatter, sav_prefix):
        x_lim = torch.max(torch.abs(x_scatter))
        x_contourf = torch.linspace(-x_lim, x_lim, 40)
        t_contourf = torch.linspace(self.t0, self.te, 50)
        v_val = torch.stack([
            self.v(t.unsqueeze(-1), x_contourf.unsqueeze(-1))
            for t in t_contourf
        ])
        v_contourf = v_val.squeeze(-1).transpose(1, 0)
        tcont_np = tensor2ndarray(t_contourf)
        xcont_np = tensor2ndarray(x_contourf)
        vcont_np = tensor2ndarray(v_contourf)
        tscat_np = tensor2ndarray(t_scatter.squeeze())
        xscat_np = tensor2ndarray(x_scatter[..., 0])

        plt.figure(figsize=(5, 3))
        plt.contourf(tcont_np, xcont_np, vcont_np, levels=30, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label("$v(t, x)$")
        plt.plot(
            tscat_np,
            xscat_np,
            marker='o',
            markersize=5,
            color='white',
            markeredgecolor='black',
            linestyle='none',
            alpha=0.6,
            label=r'Samples of $X_t$',
        )
        plt.ylabel(r'$x$')
        plt.xlabel(r'$t$')
        xlim0 = float(self.t0) - 0.025 * float(self.te - self.t0)
        xlim1 = float(self.te) + 0.025 * float(self.te - self.t0)
        plt.xlim(xlim0, xlim1)
        plt.tight_layout()
        if self.save_results is True:
            plt.savefig(f'{sav_prefix}v2dplot.pdf')
        plt.close()

    def plot_res1d(self,
                   v_approx,
                   sav_prefix,
                   ctrfunc_syspath=None,
                   numt_cutaway=3,
                   numx_cutaway=40,
                   num_scatter_path=24,
                   numdt_scatter_path=16):
        assert self.dim_x == 1

        x0_path = self.x0_points(num_scatter_path)
        if ctrfunc_syspath is None:
            t_path, xt_path = self.gen_pilpath(x0_path, numdt_scatter_path)
        else:
            t_path, xt_path, _ = self.gen_syspath(
                x0_path,
                numdt_scatter_path,
                ctrfunc_syspath,
            )
        self.plot_vtx_1d(t_path, xt_path, sav_prefix)

        t_cutaway = torch.linspace(self.t0, self.te, numt_cutaway)
        for t in t_cutaway:
            t = float(t)
            xt_max = xt_path.mean(-1, keepdims=True).abs().max()
            x_lim = 1. + (xt_max - 1.) * t / self.te
            s = torch.linspace(-x_lim, x_lim, numx_cutaway)
            x = s.unsqueeze(-1)
            self.res_on_curve([s], [x], [''],
                              f'{sav_prefix}t{t}',
                              v_approx,
                              t=t)

    def produce_results(self, v_approx, sav_prefix, ctrfunc_syspath=None):
        curve_name, s, x_s = self.xcurve_for_res(num_points=50)
        self.res_on_curve(s, x_s, curve_name, sav_prefix, v_approx)

        if self.dim_x == 1:
            self.plot_res1d(v_approx,
                            sav_prefix,
                            ctrfunc_syspath=ctrfunc_syspath)


class LinearPDE(PDEwithVtrue):
    """
    Linear parabolic PDE with v(t, x) approximated by the Feynman-Kac formula
    """

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False

    nsamp_mc = 10**6

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)

    def samp_xte_intft(self, t, x, num_dt=100, num_mc=10**4):
        dt = ((self.te - t) / num_dt)
        sqrt_dt = dt.pow(0.5)
        tn = t
        xtn = x
        int_ft = self.f(tn, xtn, None) * dt
        shape_normal = [num_mc] + list(x.shape[:-1]) + [self.dim_w]
        for _ in range(num_dt):
            normal_samp = torch.normal(mean=0.,
                                       std=1.,
                                       size=shape_normal,
                                       device=x.device)
            dwt = sqrt_dt * normal_samp
            mu_tn = self.mu_sys(tn, xtn, None)
            sgm_dwtn = self.sgm_sys(tn, xtn, None, dwt)
            tn = tn + dt
            xtn = xtn + mu_tn * dt + sgm_dwtn
            int_ft = int_ft + self.f(tn, xtn, None) * dt
        # xtn: [batch of mc, time, batch of x, dim of x]
        return xtn, int_ft

    def v(self,
          t: torch.Tensor,
          x: torch.Tensor,
          num_dt: int = 100) -> torch.Tensor:
        # Note: If self.use_dist is True, this function will utilize torch.distributed for accelerated computation.
        # Therefore, do not call this function on a single rank when self.use_dist is True.

        assert self.musys_depends_on_v is False
        assert self.sgmsys_depends_on_v is False
        assert self.f_depends_on_v is False

        assert x.ndim >= 2
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        if torch.all(t == self.te):
            return self.v_term(x)
        if self.use_dist is True:
            nsamp_mc = int(self.nsamp_mc // self.world_size) + 1
        else:
            nsamp_mc = self.nsamp_mc

        # nsamp_mc = self.nsamp_mc
        multiplier = (self.dim_w + 4 * self.dim_x) * x[..., 0].numel()
        cum_size = 0
        cum_mean = 0.
        progress = 0.
        print(f"Monte-Carlo for Ref. solution on {x.device}...")
        while cum_size < nsamp_mc:
            try:
                chunksize = get_safe_chunksize(multiplier,
                                               x.dtype,
                                               x.device,
                                               use_percent=0.3)
                chunksize = max(1, min(chunksize, nsamp_mc - cum_size))
                if self.use_dist is True:
                    # Ensure all processes use the minimum chunksize
                    chunksize_tensor = torch.tensor(chunksize, device=x.device)
                    dist.all_reduce(chunksize_tensor, op=dist.ReduceOp.MIN)
                    chunksize = int(chunksize_tensor.item())
                xte_chunk, ift_chunk = self.samp_xte_intft(
                    t,
                    x,
                    num_mc=chunksize,
                    num_dt=num_dt,
                )
                mean_vte = self.v_term(xte_chunk).mean(0)
                mean_ift = ift_chunk.mean(0)
                new_mean = mean_vte + mean_ift
                cum_size += chunksize
                if self.use_dist is True:
                    dist.all_reduce(new_mean, op=dist.ReduceOp.SUM)
                    new_mean = new_mean / self.world_size
                    rank = dist.get_rank()
                else:
                    rank = 0
                new_rate = chunksize / cum_size
                cum_mean = (1 - new_rate) * cum_mean + new_rate * new_mean

                if (cum_size / nsamp_mc > progress + 0.01) or \
                        (cum_size == nsamp_mc):
                    progress = cum_size / nsamp_mc
                    if rank == 0:
                        print(
                            f"Progress: {progress:.2%}, chunksize per rank: {chunksize}"
                        )

            except RuntimeError as err:
                if 'out of memory' in str(err):
                    free_cache(x.device)
                    chunksize = int(chunksize // 2)
                    print(
                        f"Restricted by memory of {x.device}, reduce chunksize to {chunksize}"
                    )
                    if chunksize == 0:
                        print('Memory Error: Cannot reduce chunksize=0')
                        raise err
                else:
                    raise err

        return cum_mean


class HJB(PDEwithVtrue):
    name = 'Default_HJB_equation'
    compute_cost_gap = None
    compute_cost_maxit = torch.inf

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_u = None

    def comput_cost(self, x0, u_func, num_dt=100):
        t, xt, ut = self.gen_syspath(x0, num_dt, u_func)
        f_val = self.f(t[:-1], xt[:-1], ut[:-1])
        run_cost = f_val.mean([0, 1]) * (self.te - self.t0)
        term_cost = self.v_term(xt[-1]).mean(0)
        cost = run_cost + term_cost
        if self.use_dist is True:
            dist.all_reduce(cost)
            cost = cost / self.world_size
        return cost

    def compute_cost_onx0points(self,
                                x0,
                                u_approx,
                                num_dt=100,
                                num_cost_path=64):
        # x0: [points, x]
        cost_list = []
        for x0i in x0:
            x0i_mc = torch.ones([num_cost_path, 1]) * x0i
            cost_x0i = self.comput_cost(x0i_mc, u_approx, num_dt=num_dt)
            cost_list.append(cost_x0i)
        cost_onpoints = torch.stack(cost_list, dim=0)
        return cost_onpoints

    def x0_for_cost(self, num_points):
        return self.x0_points(num_points)

    def produce_logfunc(self, v_approx, u_approx, num_cost_path=64):
        t0 = self.t0.unsqueeze(-1)
        x0_cost = self.x0_for_cost(num_cost_path)
        x0_value = self.x0_points(self.num_testpoint)

        vtrue_onx0 = self.v(t0, x0_value)
        vtrue_onx0cost = self.v(t0, x0_cost).mean()

        vtrue_ontest_l1 = torch.abs(vtrue_onx0).mean()
        vtrue_ontest_linf = torch.abs(vtrue_onx0).max()

        def log_func(it: int, _t: torch.Tensor, _xt: torch.Tensor):
            with torch.no_grad():
                err = v_approx(t0, x0_value) - vtrue_onx0
                abs_err = torch.abs(err)
                l1err = abs_err.mean() / vtrue_ontest_l1
            log = {'rel_l1err': l1err.item()}

            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()

            if self.compute_cost_gap is not None:
                if (it % self.compute_cost_gap == 0) and \
                        (it < self.compute_cost_maxit):
                    with torch.no_grad():
                        cost = self.comput_cost(x0_cost, u_approx).item()
                        cost_v0 = cost / vtrue_onx0cost.item()
                    log['cost'] = cost
                    log['vtrue_on_x0cost'] = vtrue_onx0cost.item()
                    log['cost/v0'] = cost_v0
                else:
                    log['cost'] = None
                    log['vtrue_on_x0cost'] = None
                    log['cost/v0'] = None
            return log

        return log_func


class EVP(PDEwithVtrue):

    name = 'EigenValueMeta'
    x0_for_train = {'S1': e1_curve}
    # x0_for_train = {'S2': diag_curve}
    x0pil_range = 3.

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    rate_newlamb = 0.1
    fourier_frequency = None

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.te = 10
        self.dim_w = dim_x

        self.relu6 = torch.nn.ReLU6()
        self.softplus = torch.nn.Softplus()

        self.lamb_init = 0.
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    @abstractmethod
    def set_vnn_forward(self, _vnn):
        pass

    def register_network(self, vnn: torch.nn.Module):
        self.vnn = vnn

    def get_lamb(self):
        new_lamb = self.vnn.module.lamb
        r = self.rate_newlamb
        self.lamb_val = (1 - r) * self.lamb_val.detach().item() + r * new_lamb
        return self.lamb_val

    def lamb_v(self, t, vpath):
        v_mean = vpath.mean(1, keepdim=True)
        lamb_batch = torch.log(
            (v_mean[1:] / v_mean[:1]).abs()) / (t[1:] - t[:1])
        lamb_val = lamb_batch.mean()

        r = self.rate_newlamb
        self.lamb_from_v = (
            1 - r) * self.lamb_from_v.detach().item() + r * lamb_val

        lamb_var = (lamb_batch - lamb_val).pow(2).mean()
        return lamb_val, lamb_var

    def f(self, t, _x, v):
        # lamb_val = self.get_lamb()
        with torch.no_grad():
            lamb_val = self.lamb_v(t, v)[0]
        f_val = -lamb_val * v
        return f_val

    def additional_loss(self, t_path, x_path, v_approx):
        # x0 = self.x0_points(100)
        x0 = torch.zeros((1, self.dim_x), device=self.t0.device)
        vappr_val = v_approx(None, x0)
        vtrue_val = self.v(None, x0)
        loss_val = (vtrue_val - vappr_val).mean().pow(2)

        vappr_path = v_approx(t_path, x_path)
        xnorm_mask = (x_path.norm(dim=-1, keepdim=True) > 3.)
        loss_val = loss_val + (xnorm_mask * vappr_path.pow(2)).mean()

        return loss_val

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range, self.sphere)[1]
            for (nump, c_func) in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(self.num_testpoint)
        vtrue_ontest = self.v(t0, x_test)
        vtrue_ontest_l1 = torch.abs(vtrue_ontest).mean()
        vtrue_ontest_linf = torch.abs(vtrue_ontest).max()

        def log_func(_it: int, t: torch.Tensor, xt: torch.Tensor):
            v0_approx = v_approx(t0, x_test)
            err = v0_approx - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            with torch.no_grad():
                vappr_path = v_approx(t, xt)
                lamb_from_v = self.lamb_v(t, vappr_path)[0]

            log = {
                'ev': self.lamb_val.detach().item(),
                'ev_from_v': lamb_from_v.item(),
                'ev_error': (self.true_eigenval - lamb_from_v).abs().item(),
                'rel_l1err': l1err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func
