import os
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from savresult import plot_hist

PERT_DICT = {
    '0': '0',
    '1div8': '1/8',
    '1div4': '1/4',
    '1div2': '1/2',
    '1': '1',
}


def get_csv_files(
    directdiory: str,
    filename_filter: Optional[str] = None,
) -> list:
    """
    Read all CSV file paths from the specified directory, 
    optionally filtering by a specific filename pattern.
    """
    csv_files = []
    if not os.path.exists(directdiory):
        return csv_files

    for filename in os.listdir(directdiory):
        if filename.endswith('.csv'):
            if (filename_filter is None) or (filename_filter in filename):
                csv_files.append(os.path.join(directdiory, filename))

    if len(csv_files) == 0:
        print(
            f"No CSV files found in {directdiory} with filter '{filename_filter}'"
        )
    return csv_files


def pert_from_filename(filename: str) -> float:

    pert_str = filename.split('Pert')[-1].split('_')[0]
    pert = PERT_DICT[pert_str]
    index = list(PERT_DICT.keys()).index(pert_str)
    return pert, index


def parse_resonline_eps(csv_list: List[str]) -> Dict[str, float]:
    index = []
    labels = []
    coord = []
    vtrue = []
    vappr = []
    for csv in csv_list:
        df = pd.read_csv(csv)
        pert_val, idx = pert_from_filename(os.path.basename(csv))
        index.append(idx)
        labels.append(r'$\epsilon = ' + f'{pert_val}$')
        coord.append(df['# coord_S2'].values)
        vtrue.append(df['vtrue_S2'].values)
        vappr.append(df['vappr_S2'].values)

    v_list = [vtrue[0]] + [vappr[i] for i in index]
    labels = ['True'] + [labels[i] for i in index]
    return coord[0], v_list, labels


def plot_on_curve(
        s,
        v_list,
        sav_prefix,
        labels=None,
        ylim=None,
        alpha=0.85,
        figsize=(6, 3.5),
        legend_loc='upper left',
        bbox_to_anchor=(1.05, 1),
        markers=None,
        linestyle_of_true='-',
):
    num_v = len(v_list)
    if labels is None:
        labels = [None] * num_v
    cmap = plt.get_cmap('tab10')(np.arange(num_v))
    if markers is None:
        markers = cycle(('x', '+', 's', 'D', '^', 'v', 'o'))

    plt.figure(figsize=figsize)
    for i in range(num_v):
        if labels[i] == 'True':
            plt.plot(
                s,
                v_list[i],
                label=labels[i],
                color=cmap[i],
                marker=None,
                linestyle=linestyle_of_true,
            )
        else:
            plt.scatter(
                s,
                v_list[i],
                label=labels[i],
                color=cmap[i],
                marker=next(markers),
                s=25,
                alpha=alpha,
            )

    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_loc)
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(sav_prefix, dpi=600)
    plt.close()


def plot_perturb():
    # 绘制关于 HJB 方程包含 perturbation 项的数值结果

    dir = './m140m141_hc1a_pert/outputs'
    # Change this to your directory containing CSV files
    csv_files = get_csv_files(dir, filename_filter='res_on_line')

    coord, v_list, labels = parse_resonline_eps(csv_files)
    plot_on_curve(
        coord,
        v_list,
        sav_prefix='./hjb1a_pert.pdf',
        labels=labels,
    )


def read_csv_without_wellsign(path):
    with open(path, 'r') as f:
        columns = f.readline().strip().replace('# ', '').split(',')
    return pd.read_csv(path, skiprows=1, header=None, names=columns)


def compare_mscale_dnn():
    figsize = (5, 3.5)
    nw_to_dir = {
        'Standard DNN': './result/m164_hc2-rpt5/outputs',
        'MscaleDNN': './result/m169_hc2_Mscale_S2S3/outputs',
    }
    nw_to_linedf = {}
    nw_to_logdf = {}
    for nw, dir in nw_to_dir.items():
        csv_line = get_csv_files(dir, filename_filter='r0_res_on_line')[0]
        csv_log = get_csv_files(dir, filename_filter='r0_log')[0]

        df_line = read_csv_without_wellsign(csv_line)
        df_log = read_csv_without_wellsign(csv_log)
        nw_to_linedf[nw] = df_line
        nw_to_logdf[nw] = df_log

    labels = list(nw_to_dir.keys())
    for line in ['S2', 'S3']:
        coord = nw_to_linedf[labels[0]][f'coord_{line}'].values
        vtrue = nw_to_linedf[labels[0]][f'vtrue_{line}'].values
        vappr = [
            nw_to_linedf[k][f'vappr_{line}'].values
            for k in nw_to_linedf.keys()
        ]
        plot_on_curve(coord, [vtrue] + vappr,
                      sav_prefix=f'./Plot_res_on_{line}.pdf',
                      labels=['True'] + labels,
                      figsize=figsize,
                      legend_loc='upper center',
                      bbox_to_anchor=None,
                      alpha=1.0,
                      markers=cycle(('v', 'X')),
                      linestyle_of_true='-')

    iter_steps = [nw_to_logdf[k]['it'].values for k in nw_to_dir.keys()]
    linf_err = [nw_to_logdf[k]['rel_linferr'].values for k in nw_to_dir.keys()]
    color_list = plt.get_cmap('tab10')(np.arange(1, len(iter_steps) + 1))
    plot_hist('./Plot_linf_err.pdf',
              iter_steps,
              linf_err,
              lable_list=labels,
              figsize=figsize,
              color_list=color_list)


if __name__ == "__main__":
    # plot_perturb()
    compare_mscale_dnn()
