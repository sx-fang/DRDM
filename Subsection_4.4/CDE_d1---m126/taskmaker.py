# Produce task files, which will be used by runtask.py

from itertools import product

from runtask import DEFAULT_CONFIG, TASK_PATH, get_config, sav_config


def maxit_default(dim_x):
    if dim_x <= 1000:
        maxit = 2000
    elif dim_x <= 5000:
        maxit = 4000
    else:
        maxit = 6000
    return maxit


def maxit_long(dim_x):
    if dim_x <= 1000:
        maxit = 3000
    elif dim_x <= 5000:
        maxit = 6000
    else:
        maxit = 9000
    return maxit


def batsize_default(dim_x):
    if dim_x <= 1000:
        bsz = 256
    elif dim_x <= 5000:
        bsz = 128
    elif dim_x <= 10000:
        bsz = 64
    else:
        bsz = 32
    return bsz


def eposize_default(dim_x):
    if dim_x <= 5000:
        esz = 10000
    elif dim_x <= 10000:
        esz = 5000
    else:
        esz = 2000
    return esz


def default_lrfunc(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-3 / dim_x**expo
    lr0_v = 3 * 1e-3 / dim_x**expo
    lr0_rho = 3 * 1e-2 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def small_lrfunc(dim_x):
    lr0_u, lr0_v, lr0_rho = default_lrfunc(dim_x)
    return lr0_u * 0.1, lr0_v * 0.1, lr0_rho * 0.1


def ev_lrfunc(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-4 / dim_x**expo
    lr0_v = 3 * 1e-4 / dim_x**expo
    lr0_rho = 3 * 1e-3 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def var_width(dim_x):
    if dim_x <= 100:
        wid = 5 * dim_x + 10
    elif dim_x <= 1000:
        wid = 2 * dim_x + 10
    else:
        wid = dim_x + 10
    wid = max(wid, 100)
    return wid


def make_task(
    dimx_list,
    example_list,
    numdt_list,
    num_hidd=4,
    start_idx=0,
    lr_func=default_lrfunc,
    maxit_func=None,
    width_func=var_width,
    repeat_time=1,
    syspath_as_pilpath=True,
    batsize_func=batsize_default,
    eposize_func=eposize_default,
    enable_autocast=False,
):
    idx = start_idx
    tot_idx = start_idx - 1 + len(dimx_list) * len(example_list)
    length = len(str(tot_idx))

    if maxit_func is None:
        maxit_func = maxit_default

    for dimx, exmaple, num_dt in product(dimx_list, example_list, numdt_list):
        bsz = batsize_func(dimx)
        esz = eposize_func(dimx)
        wid = width_func(dimx)
        config = get_config(DEFAULT_CONFIG)
        config.set('Environment', 'enable_autocast', str(enable_autocast))
        config.set('Example', 'dim_x', str(dimx))
        config.set('Network', 'width_u', str(wid))
        config.set('Network', 'width_v', str(wid))
        config.set('Network', 'num_hidden_u', str(num_hidd))
        config.set('Network', 'num_hidden_v', str(num_hidd))

        config.set('Example', 'name', exmaple)
        config.set('Example', 'repeat_time', str(repeat_time))

        lr0_u, lr0_v, lr0_rho = lr_func(dimx)
        config.set('Optimizer', 'lr0_u', str(lr0_u))
        config.set('Optimizer', 'lr0_v', str(lr0_v))
        config.set('Optimizer', 'lr0_rho', str(lr0_rho))

        max_it = maxit_func(dimx)
        config.set('Training', 'num_dt', str(num_dt))
        config.set('Training', 'max_iter', str(max_it))
        config.set('Training', 'batsize', str(bsz))
        config.set('Training', 'epochsize', str(esz))
        config.set('Training', 'syspath_as_pilpath', str(syspath_as_pilpath))
        idx_str = str(idx).zfill(length)
        sav_name = f'{idx_str}_{exmaple}_d{dimx}_W{wid}_H{num_hidd}_N{num_dt}'
        sav_config(config, f'{TASK_PATH}/{sav_name}')
        idx = idx + 1


def main():

    # maxit_func = maxit_long
    maxit_func = lambda _d: 9000
    # maxit_func = maxit_default
    # maxit_func = lambda _d: 6000
    # width_func = lambda _d: 10000
    width_func = lambda _d: 100
    # width_func = var_width
    num_hidd = 6
    lr_func = default_lrfunc
    # lr_func = small_lrfunc
    batsize_func = batsize_default
    eposize_func = eposize_default
    syspath_as_pilpath = False
    enable_autocast = False

    repeat_time = 1
    # dimx_list = [1]
    dimx_list = [1, 2, 3, 4, 5]
    numdt_list = [100]
    # numdt_list = [int(8 * (128 / 8)**(i / 10)) for i in range(1, 11)]
    # dimx_list = [100, 1000]
    exa_list = [
        # 'Shock1a',
        'Shock1aBmPil',
        # 'Shock1b',
        'Shock1bBmPil',
        # 'HjbHopfCole1a',
        # 'HjbHopfColeSumG',
        # 'HjbHopfCole1aBmPil',
        # 'HjbHopfCole1b',
        # 'HjbHopfCole1c',
        # 'HjbHopfCole2',
        # 'HjbHopfCole2BmPil',
        # 'HjbHopfCole4a',
        # 'HjbHopfCole4aBmPil',
        # 'HjbHopfCole4b',
        'HjbHopfCole4bBmPil',
        # 'QLP1',
        # 'QLP2a',
        # 'QLP2b',
        # 'QLP1_UnitBall',
        # 'QLP1_varci',
        # 'QLP2a_varci',
        # 'QLP2b_varci',
    ]

    make_task(
        dimx_list,
        exa_list,
        numdt_list,
        num_hidd=num_hidd,
        maxit_func=maxit_func,
        width_func=width_func,
        repeat_time=repeat_time,
        lr_func=lr_func,
        syspath_as_pilpath=syspath_as_pilpath,
        batsize_func=batsize_func,
        eposize_func=eposize_func,
        enable_autocast=enable_autocast,
    )


if __name__ == "__main__":
    main()
