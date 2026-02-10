import os
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    return csv_files


def pert_from_filename(filename: str) -> float:

    pert_str = filename.split('Pert')[-1].split('_')[0]
    pert = PERT_DICT[pert_str]
    index = list(PERT_DICT.keys()).index(pert_str)
    return pert, index


def parse_resonline(csv_list: List[str]) -> Dict[str, float]:
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
):
    num_v = len(v_list)
    if labels is None:
        labels = [None] * num_v
    cmap = plt.get_cmap('tab10')(np.arange(num_v))
    markers = cycle(('x', '+', 's', 'D', '^', 'v', 'o'))
    plt.figure(figsize=(6, 3.5))
    for i in range(num_v):
        if labels[i] == 'True':
            plt.plot(
                s,
                v_list[i],
                label=labels[i],
                color=cmap[i],
                marker=None,
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(sav_prefix, dpi=600)
    plt.close()


def main():
    dir = './m140m141_hc1a_pert/outputs'  # Change this to your directory containing CSV files
    csv_files = get_csv_files(dir, filename_filter='res_on_line')

    coord, v_list, labels = parse_resonline(csv_files)
    plot_on_curve(
        coord,
        v_list,
        sav_prefix='./hjb1a_pert.pdf',
        labels=labels,
    )
    pass


if __name__ == "__main__":
    main()
