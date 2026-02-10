# The core function of this script is run_task.
import ast
import os
import random
import socket
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Dict, Optional
import math

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import ex_bvp
import ex_evp
import ex_hjb_constantceff
import ex_hjb_hopfcole
import ex_miscellany
import ex_pde_linear
import ex_qlp
import martnetdf
from ex_bvp import PBE1
from ex_evp import EVP
from ex_meta import HJB, PDE, PDEwithVtrue
from martnetdf import PathSampler, train_martnet
from networks import make_multi_scale_net, make_var_scale_net, DNNtx, DNNx, EigenFuncValue
from savresult import (plot_hist_summary, plot_path_summary, save_hist,
                       save_pathres, summary_hist, summary_repath)

DEFAULT_CONFIG = './default_config.ini'
TASK_PATH = './taskfiles'

# Please ensure that class names in the modules listed in EXAMPLE_MODULES are unique.
EXAMPLE_MODULES = (
    ex_hjb_hopfcole,
    ex_hjb_constantceff,
    ex_miscellany,
    ex_pde_linear,
    ex_evp,
    ex_qlp,
    ex_bvp,
)


def get_classes(module):
    # Get all class names and objects in the module
    classes = {}
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and obj.__module__ == module.__name__:
            classes[name] = obj
    return classes


def get_example_classes() -> Dict['str', type]:
    """Collects and returns all custom classes from the example modules in EXAMPLE_MODULES while ensuring no name conflicts."""

    # 用于记录类名与模块的对应关系
    class_to_modules = {}

    # 用于注册所有类的全局字典
    class_registry = {}

    # 遍历所有模块，收集类并记录来源
    for module in EXAMPLE_MODULES:
        module_name = module.__name__
        classes = get_classes(module)
        for class_name, cls in classes.items():
            if class_name not in class_to_modules:
                class_to_modules[class_name] = []
            class_to_modules[class_name].append(module_name)
            class_registry[class_name] = cls

    # 检查类名重复
    duplicate_classes = {
        cls: modules
        for cls, modules in class_to_modules.items() if len(modules) > 1
    }

    if duplicate_classes:
        exmod_names = [mod.__name__ for mod in EXAMPLE_MODULES]
        error_msg = f"Ensure that class names in the modules {exmod_names} are unique. \n"
        for cls_name, module_list in duplicate_classes.items():
            error_msg += f"- Class '{cls_name}' found in modules: {', '.join(module_list)}\n"
        raise ValueError(error_msg)
    return class_registry


def set_seed(config, rank: int = 0):
    seed_str = config.get('Environment', 'seed')
    if seed_str == "None":
        return None
    else:
        seed = ast.literal_eval(seed_str) + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.xpu.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def clean_mp():
    dist.barrier()
    dist.destroy_process_group()


def get_config(path):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path, encoding='utf-8')
    return config


def sav_config(config, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(f'{path}.ini', 'w') as configfile:
        config.write(configfile)


def set_torchdtype(config):
    dtype = config['Environment']['torch_dtype']
    if dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    elif dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float16':
        torch.set_default_dtype(torch.float16)
    else:
        raise ValueError(f"Unsupported torch.dtype: {dtype}")
    # print(f"Default torch dtype set to: {torch.get_default_dtype()}")
    # print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")


def print_gpu():
    for rank in range(torch.cuda.device_count()):
        gpu_i = torch.device(f"cuda:{rank}")
        properties = torch.cuda.get_device_properties(gpu_i)
        print(f"GPU {rank}: {properties.name}")
        print("Memory: ", properties.total_memory)
    for rank in range(torch.xpu.device_count()):
        gpu_i = torch.device(f"xpu:{rank}")
        properties = torch.xpu.get_device_properties(gpu_i)
        print(f"GPU {rank}: {properties.name}")
        print("Memory: ", properties.total_memory)


def parse_device(config):
    dev_conf = config['Environment']['device']
    n_cuda = torch.cuda.device_count()
    n_xpu = torch.xpu.device_count()

    if dev_conf == 'default':
        if n_cuda > 0:
            device = 'cuda'
        elif n_xpu > 0:
            device = 'xpu'
        else:
            device = 'cpu'
    elif dev_conf in ('cuda', 'xpu') and (max(n_cuda, n_xpu) == 0):
        warnings.warn('GPU is unavailable. Use cpu instead.')
        device = 'cpu'
    elif dev_conf == 'cuda' and (n_cuda > 0):
        device = 'cuda'
    elif dev_conf == 'xpu' and (n_xpu > 0):
        device = 'xpu'
    elif dev_conf == 'cpu':
        device = 'cpu'
    else:
        raise ValueError(
            f"Invalid config setting: config[Environment][device] = {dev_conf}"
        )

    if device in ('cuda', 'xpu'):
        print_gpu()
        n_gpu = n_cuda if device == 'cuda' else n_xpu
        ws_str = config['Environment']['world_size']
        if ws_str == 'auto':
            world_size = n_gpu
        else:
            world_size = int(ws_str)
            if world_size > n_gpu:
                world_size = n_gpu
                warnings.warn(
                    f'Available GPUs is less than world_size = {world_size}. \n Reset world_size = {n_gpu}.'
                )
    else:
        world_size = 1
    return device, world_size


def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = random.randint(1024, 65535)
    while True:
        try:
            sock.bind(('localhost', port))
            break
        except socket.error:
            port = random.randint(1024, 65535)
    sock.close()
    return port


def init_processgp(rank, world_size):
    try:
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)
        if rank == 0:
            print('Using nccl backend')
    except RuntimeError:
        dist.init_process_group(backend="gloo",
                                rank=rank,
                                world_size=world_size)
        if rank == 0:
            print('Nccl is unavailable. Use gloo backend instead.')


def set_master(config):
    mp_str = config.get('Environment', 'master_port')
    if mp_str == 'random':
        master_port = get_free_port()
    else:
        master_port = int(mp_str)
    use_libuv = config.get('Environment', 'use_libuv')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ["USE_LIBUV"] = use_libuv
    print(f'Master Port: {master_port}')


def parse_example(config, use_dist) -> PDE:
    exa_name = config.get('Example', 'name')
    dim_x = config.getint('Example', 'dim_x')

    try:
        example_class = get_example_classes()[exa_name]
    except KeyError:
        raise ValueError(
            f"Example '{exa_name}' not found in modules {[mod.__name__ for mod in EXAMPLE_MODULES]}."
        )
    example = example_class(dim_x, use_dist=use_dist)
    return example


def parse_martnet(config, example: PDE, use_dist=False, rank=0):

    vnn, optim_v, sch_v = parse_v(config,
                                  example,
                                  use_dist=use_dist,
                                  rank=rank)
    rhonn, optim_asc, sch_rho = parse_rho(config,
                                          example,
                                          use_dist=use_dist,
                                          rank=rank)
    if isinstance(example, HJB):
        unn, optim_u, sch_u = parse_u(config,
                                      example,
                                      use_dist=use_dist,
                                      rank=rank)
        nets = (unn, vnn, rhonn)
        optim_desc = (optim_u, optim_v)
        schs = (sch_u, sch_v, sch_rho)
        meth_str = config.get('Example', 'method_soc')
        num_cost_path = config.getint('Example', 'num_cost_path')
        if use_dist is True:
            num_cost_path = max(1, int(num_cost_path // dist.get_world_size()))

        log_func = example.produce_logfunc(
            vnn,
            unn,
            num_cost_path=num_cost_path,
        )
    else:
        nets = (vnn, rhonn)
        log_func = example.produce_logfunc(vnn)
        if isinstance(example, EVP):
            meth_str = config.get('Example', 'method_evp')
            optim_lamb, sch_lamb = parse_lamb(config,
                                              vnn.module.eigenval_parameters())
            optim_desc = (optim_v, optim_lamb)
            schs = (sch_v, sch_lamb, sch_rho)
        else:
            meth_str = config.get('Example', 'method_pde')
            optim_desc = optim_v
            schs = (sch_v, sch_rho)

    MartNet = getattr(martnetdf, meth_str)
    martnet = MartNet(example,
                      nets,
                      num_dt=config.getint('Training', 'num_dt'),
                      use_dist=use_dist,
                      rank=rank)

    kwarg_train = {
        "schs": schs,
        "max_iter": config.getint('Training', 'max_iter'),
        "step_desc": config.getint('Training', 'inner_step_descend'),
        "step_asc": config.getint('Training', 'inner_step_ascend'),
        "log_func": log_func,
        "enable_scaler": config.getboolean('Environment', 'enable_scaler'),
    }
    return martnet, optim_desc, optim_asc, kwarg_train


def parse_pathsampler(config,
                      example,
                      ctr_func,
                      world_size=1,
                      rank=0) -> callable:
    epochsize = config.getint('Training', 'epochsize')
    epochsize = max(1, int(epochsize // world_size))
    ip_time_gap = 1.0 if world_size > 1 else 0.0

    rate_newpath = config.getfloat('Training', 'rate_newpath')
    num_dt = config.getint('Training', 'num_dt')
    path_sampler = PathSampler(example,
                               epochsize,
                               num_dt=num_dt,
                               rank=rank,
                               ctr_func=ctr_func,
                               rate_newpath=rate_newpath,
                               ip_time_gap=ip_time_gap)
    return path_sampler.get_pathbat


def nets2dpp(nets, rank):
    return [DDP(nn.to(rank), device_ids=[rank]) for nn in nets]


def str2dtype(dtype_str: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype."""
    if dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float64':
        return torch.float64
    elif dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def parse_scale_net(config, num_units, pde, kwargs_dnn):
    varscale_lb = torch.tensor(config.getfloat('Network', 'varscale_lb'))
    varscale_ub = torch.tensor(config.getfloat('Network', 'varscale_ub'))
    width_bound = varscale_ub - varscale_lb

    shell_func = lambda _x, y: varscale_lb + torch.sigmoid(y) * width_bound

    if isinstance(pde, PBE1) or isinstance(pde, EVP):
        scale_net = DNNx(num_units, shell_func=shell_func, **kwargs_dnn)
    else:
        scale_net = DNNtx(num_units, shell_func=shell_func, **kwargs_dnn)

    return scale_net


def parse_u(config, pde: HJB, use_dist=False, rank=0):
    num_subnets_u = config.getint('Network', 'num_subnets_u')
    multiscale_u = config.getboolean('Network', 'multiscale_u')
    scale_step_u = eval(config.get('Network', 'scale_step_u'))

    width_u = eval(config.get('Network', 'width_u'))
    if multiscale_u:
        width_u = math.ceil(width_u / num_subnets_u)

    act_u = getattr(nn, config.get('Network', 'act_u'))
    num_hidden_u = config.getint('Network', 'num_hidden_u')

    enable_autocast = config.getboolean('Environment', 'enable_autocast')
    autocast_dtype = str2dtype(config.get('Environment', 'autocast_dtype'))
    layer_norm = config.getboolean('Network', 'layer_norm')

    dims_dnntx = [pde.dim_x] + [width_u] * num_hidden_u + [pde.dim_u]
    kwargs_dnntx = {
        'act_func': act_u,
        'enable_autocast': enable_autocast,
        'autocast_dtype': autocast_dtype,
        'layer_norm': layer_norm,
    }
    unn_generator = lambda: DNNtx(dims_dnntx, **kwargs_dnntx)
    if multiscale_u:
        unn = make_multi_scale_net(unn_generator,
                                   num_subnets_u,
                                   step_scale=scale_step_u)
    else:
        unn = unn_generator()

    varscale = config.getboolean('Network', 'varscale')
    if varscale:
        num_units = [pde.dim_x] + [width_u] * num_hidden_u + [1]
        scale_net = parse_scale_net(config, num_units, pde, kwargs_dnntx)
        unn = make_var_scale_net(unn, scale_net)

    if use_dist is True:
        unn = DDP(unn.to(rank), device_ids=[rank])

    optname_u = config.get('Optimizer', 'optimizer_u')
    kwargs_u = ast.literal_eval(config['Optimizer']['kwargs_u'])
    lr0_u = eval(config.get('Optimizer', 'lr0_u'))
    optim_u = getattr(torch.optim, optname_u)(unn.parameters(),
                                              lr=lr0_u,
                                              **kwargs_u)

    decay_gap_u = config.getint('Optimizer', 'decay_stepgap_u')
    decay_rate_u = eval(config.get('Optimizer', 'decay_rate_u'))
    sch_u = torch.optim.lr_scheduler.StepLR(optim_u,
                                            step_size=decay_gap_u,
                                            gamma=decay_rate_u)
    return unn, optim_u, sch_u


def parse_v(config, pde: PDE, use_dist=False, rank=0):
    num_subnets_v = config.getint('Network', 'num_subnets_v')
    multiscale_v = config.getboolean('Network', 'multiscale_v')
    scale_step_v = eval(config.get('Network', 'scale_step_v'))

    width_v = eval(config.get('Network', 'width_v'))
    if multiscale_v:
        width_v = math.ceil(width_v / num_subnets_v)

    act_v = getattr(nn, config.get('Network', 'act_v'))
    num_hidden_v = config.getint('Network', 'num_hidden_v')
    num_units = [pde.dim_x] + [width_v] * num_hidden_v + [1]

    enable_autocast = config.getboolean('Environment', 'enable_autocast')
    autocast_dtype = str2dtype(config.get('Environment', 'autocast_dtype'))
    layer_norm = config.getboolean('Network', 'layer_norm')

    kwargs_dnn = {
        'act_func': act_v,
        'enable_autocast': enable_autocast,
        'autocast_dtype': autocast_dtype,
        'layer_norm': layer_norm,
    }

    if isinstance(pde, EVP):

        def vnn_generator():
            nn = EigenFuncValue(num_units,
                                init_lamb=pde.lamb_init,
                                fourier_frequency=pde.fourier_frequency,
                                **kwargs_dnn)
            return pde.set_vnn_forward(nn)

    elif isinstance(pde, PBE1):
        vnn_generator = lambda: DNNx(num_units, **kwargs_dnn)
    else:
        vnn_generator = lambda: DNNtx(num_units, **kwargs_dnn)
    if multiscale_v:
        vnn = make_multi_scale_net(vnn_generator,
                                   num_subnets_v,
                                   step_scale=scale_step_v)
    else:
        vnn = vnn_generator()

    varscale = config.getboolean('Network', 'varscale')
    if varscale:
        num_units = [pde.dim_x] + [width_v] * num_hidden_v + [1]
        scale_net = parse_scale_net(config, num_units, pde, kwargs_dnn)
        vnn = make_var_scale_net(vnn, scale_net)

    if use_dist is True:
        vnn = DDP(vnn.to(rank), device_ids=[rank])

    params = vnn.module.eigenfunc_parameters() if isinstance(
        pde, EVP) else vnn.parameters()

    optname_v = config.get('Optimizer', 'optimizer_v')
    kwargs_v = ast.literal_eval(config['Optimizer']['kwargs_v'])
    lr0_v = eval(config.get('Optimizer', 'lr0_v'))
    decay_gap_v = config.getint('Optimizer', 'decay_stepgap_v')
    decay_rate_v = eval(config.get('Optimizer', 'decay_rate_v'))

    optim_v = getattr(torch.optim, optname_v)(params, lr=lr0_v, **kwargs_v)
    sch_v = torch.optim.lr_scheduler.StepLR(optim_v,
                                            step_size=decay_gap_v,
                                            gamma=decay_rate_v)
    return vnn, optim_v, sch_v


def parse_lamb(config, lamb_params: nn.Parameter):
    optname_lamb = config.get('Optimizer', 'optimizer_lamb')
    kwargs_lamb = ast.literal_eval(config['Optimizer']['kwargs_lamb'])
    lr0_lamb = eval(config.get('Optimizer', 'lr0_lamb'))
    optim_lamb = getattr(torch.optim, optname_lamb)(lamb_params,
                                                    lr=lr0_lamb,
                                                    **kwargs_lamb)
    decay_gap_lamb = config.getint('Optimizer', 'decay_stepgap_lamb')
    decay_rate_lamb = eval(config.get('Optimizer', 'decay_rate_lamb'))
    sch_lamb = torch.optim.lr_scheduler.StepLR(optim_lamb,
                                               step_size=decay_gap_lamb,
                                               gamma=decay_rate_lamb)
    return optim_lamb, sch_lamb


def parse_rho(config, pde: PDE, use_dist=False, rank=0):
    num_subnets_rho = config.getint('Network', 'num_subnets_rho')
    multiscal_rho = config.getboolean('Network', 'multiscale_rho')
    scale_step_rho = eval(config.get('Network', 'scale_step_rho'))

    width_rho = eval(config.get('Network', 'width_rho'))
    if multiscal_rho:
        width_rho = math.ceil(width_rho / num_subnets_rho)

    act_rho = getattr(nn, config.get('Network', 'act_rho'))
    num_hidden_rho = config.getint('Network', 'num_hidden_rho')
    num_units = [pde.dim_x] + [width_rho] * (num_hidden_rho + 1)

    rho_shell_str = config.get('Network', 'rho_shell')
    rho_shell = getattr(torch, rho_shell_str)
    scale_factor_rho = config.getfloat('Network', 'scale_factor_rho')

    kwargs_dnn = {
        'act_func': act_rho,
        'enable_autocast': False,
        'layer_norm': False,
    }
    shell_test = lambda _x, y: rho_shell(y)
    rho_generator = lambda: DNNtx(num_units,
                                  shell_func=shell_test,
                                  scale_factor=scale_factor_rho,
                                  **kwargs_dnn)

    if multiscal_rho:
        rhonn = make_multi_scale_net(rho_generator,
                                     num_subnets_rho,
                                     step_scale=scale_step_rho)
    else:
        rhonn = rho_generator()

    # varscale = config.getboolean('Network', 'varscale')
    # if varscale:
    #     num_units = [pde.dim_x] + [width_rho] * num_hidden_rho + [1]
    #     scale_net = parse_scale_net(config, num_units, pde, kwargs_dnn)
    #     rhonn = make_var_scale_net(rhonn, scale_net)

    if use_dist is True:
        rhonn = DDP(rhonn.to(rank), device_ids=[rank])

    optname_rho = config.get('Optimizer', 'optimizer_rho')
    kwargs_rho = ast.literal_eval(config['Optimizer']['kwargs_rho'])
    lr0_rho = eval(config.get('Optimizer', 'lr0_rho'))
    decay_gap_rho = config.getint('Optimizer', 'decay_stepgap_rho')
    decay_rate_rho = eval(config.get('Optimizer', 'decay_rate_rho'))

    optim_rho = getattr(torch.optim, optname_rho)(
        rhonn.parameters(),
        lr=lr0_rho,
        **kwargs_rho,
    )
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho,
                                              step_size=decay_gap_rho,
                                              gamma=decay_rate_rho)

    return rhonn, optim_rho, sch_rho


def solve(config, use_dist, rank, sav_name=None):

    batsize = config.getint('Training', 'batsize')
    if use_dist is True:
        wd_size = dist.get_world_size()
        batsize = max(1, int(batsize // wd_size))
        ip_time_gap = 1.0  # Prevent the output file from becoming excessively large when running on SLURM
    else:
        wd_size = 1
        ip_time_gap = 0.

    example = parse_example(config, use_dist)
    martnet, optim_desc, optim_asc, kwarg_train = parse_martnet(
        config,
        example,
        use_dist=use_dist,
        rank=rank,
    )
    syspath_as_pilpath = config.getboolean('Training', 'syspath_as_pilpath')
    if syspath_as_pilpath is True:
        ctr_func = martnet.unn if isinstance(example, HJB) else martnet.vnn
    else:
        ctr_func = None
    pathsamp_func = parse_pathsampler(config,
                                      example,
                                      ctr_func,
                                      world_size=wd_size,
                                      rank=rank)
    hist_dict = train_martnet(martnet,
                              pathsamp_func,
                              optim_desc,
                              optim_asc,
                              batsize=batsize,
                              ip_time_gap=ip_time_gap,
                              **kwarg_train)

    path_dict = None
    if sav_name is not None:
        output_dir = config.get('Environment', 'output_dir')
        sav_path = f"{output_dir}/{sav_name}"

        if isinstance(example, PDEwithVtrue):
            example.produce_results(martnet.vnn,
                                    sav_path,
                                    ctrfunc_syspath=ctr_func)
            path_dict = example.res_on_path(martnet.vnn,
                                            ctrfunc_syspath=ctr_func)

        if (rank == 0) or (use_dist is False):
            save_hist(hist_dict, sav_path)
            if path_dict is not None:
                save_pathres(path_dict, sav_path)
    return hist_dict, path_dict


def task_on_gpu(rank: int,
                gpu_device: str,
                world_size: int,
                config,
                return_dict,
                sav_name: Optional[str] = None):
    assert gpu_device in ('cuda', 'xpu')
    torch.set_default_device(f'{gpu_device}:{rank}')
    use_dist = world_size > 1
    if use_dist is True:
        init_processgp(rank, world_size)
    set_torchdtype(config)
    set_seed(config, rank=rank)

    rep_time = config.getint('Example', 'repeat_time')
    issav_everytime = config.getboolean(
        'Example',
        'save_result_for_every_repeat_time',
    )
    hdict_list = []
    rdict_list = []
    for r in range(rep_time):
        if (issav_everytime is True) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        elif (r == 0) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        else:
            savname_r = None
        hist_dict, res_dict = solve(config, use_dist, rank, sav_name=savname_r)
        torch.cuda.empty_cache()
        hdict_list.append(hist_dict)
        rdict_list.append(res_dict)
    if rank == 0:
        hist_summ = summary_hist(hdict_list)
        return_dict[rank] = hist_summ
        if sav_name is not None:
            output_dir = config.get('Environment', 'output_dir')
            sav_path = f"{output_dir}/{sav_name}_"
            sav_config(config, sav_path)
            hist_summ.to_csv(f'{sav_path}summary_hist.csv')
            plot_hist_summary(hist_summ, sav_path)

            has_resdict = any(rdict is not None for rdict in rdict_list)
            if has_resdict:
                path_summ = summary_repath(rdict_list)
                path_summ.to_csv(f'{sav_path}summary_path.csv')
                plot_path_summary(path_summ, sav_path)

    if use_dist is True:
        clean_mp()


def task_on_cpu(config, sav_name: str):
    set_torchdtype(config)
    set_seed(config)
    torch.set_default_device('cpu')

    hdict_list = []
    rdict_list = []
    rep_time = config.getint('Example', 'repeat_time')
    issav_everytime = config.getboolean('Example',
                                        'save_result_for_every_repeat_time')
    for r in range(rep_time):
        if (issav_everytime is True) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        elif (r == 0) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        else:
            savname_r = None
        hist_dict, res_dict = solve(config, False, None, sav_name=savname_r)
        hdict_list.append(hist_dict)
        rdict_list.append(res_dict)

    hist_summ = summary_hist(hdict_list)
    if sav_name is not None:
        output_dir = config.get('Environment', 'output_dir')
        sav_path = f"{output_dir}/{sav_name}_"
        sav_config(config, sav_path)
        hist_summ.to_csv(f'{sav_path}summary_hist.csv')
        plot_hist_summary(hist_summ, sav_path)

        has_resdict = any(rdict is not None for rdict in rdict_list)
        if has_resdict:
            path_summ = summary_repath(rdict_list)
            path_summ.to_csv(f'{sav_path}summary_path.csv')
            plot_path_summary(path_summ, sav_path)
    return hist_summ


def run_task(config, sav_name=None) -> pd.DataFrame:
    device, world_size = parse_device(config)
    if device in ('cuda', 'xpu'):
        print(f'Used device: {device} * {world_size}')
        if world_size == 1:
            return_dict = dict()
            task_on_gpu(0,
                        device,
                        world_size,
                        config,
                        return_dict,
                        sav_name=sav_name)
        else:
            set_master(config)
            return_dict = mp.Manager().dict()
            mp.spawn(task_on_gpu,
                     args=(device, world_size, config, return_dict, sav_name),
                     nprocs=world_size,
                     join=True)
        hist_df = return_dict[0]
    else:
        print(f'Used device: {device}')
        hist_df = task_on_cpu(config, sav_name=sav_name)
    return hist_df


def findfiles(base):
    for root, _, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_taskfiles(path):
    taskfiles = [f for f in findfiles(path) if f.split('.')[-1] == 'ini']
    taskfiles.sort()
    return taskfiles


def main():
    if not os.path.isdir(TASK_PATH):
        os.mkdir(TASK_PATH)
    config_file = get_taskfiles(TASK_PATH)
    if len(config_file) == 0:
        config = get_config(DEFAULT_CONFIG)
        sav_name = Path(DEFAULT_CONFIG).stem
        run_task(config, sav_name=sav_name)
    else:
        for file in config_file:
            print(f'Task starts: {file}')
            config = get_config(file)
            sav_name = Path(file).stem
            run_task(config, sav_name=sav_name)


if __name__ == '__main__':
    main()
