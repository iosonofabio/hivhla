import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def filter_ndigit(prds, ndigit):
    return prds.loc[{'hla': [x for x in prds['hla'].values if len(x) == ndigit + 1]}]


def load_predictions(fname=None, fmt='xarray'):
    if fname is None:
        fname = 'data/from_Jonathan/All_hlaProb.txt'

    if fmt == 'xarray':
        prds = pd.read_csv(fname, sep='\t', index_col=[0, 1])
        prds = (xr.DataArray(prds)
                  .unstack('dim_0')
                  .rename({'dim_1': 'probability'})
                  .transpose('hla', 'pid', 'probability'))
        prds.coords['probability'] = ['posterior', 'prior']

        return prds

    prds = {}
    with open(fname) as ifile:
        ifile.readline()
        for line in ifile:
            label, HLA, posterior, prior = line.strip().split()
            # label is e.g. p4_3156 -> (p4, 3156)
            label = (label.split('_')[0], int(label.split('_')[1]))
            posterior = float(posterior)
            prior = float(prior)

            if label not in prds:
                prds[label] = {}
            prds[label][HLA] = (posterior, prior)

    return prds


def load_HLA(fname):
    HLAs = {}
    with open(fname) as ifile:
        for line in ifile:
            entries = line.rstrip('\n').strip().split(',')
            pat = entries[0]
            HLAs[pat] = {}
            HLAs[pat]['A'] = [x for x in entries[1:3] if x]
            HLAs[pat]['B'] = [x for x in entries[3:5] if x]
            HLAs[pat]['C'] = [x for x in entries[5:7] if x]
    return HLAs


def plot_heatmap(prds=None, hlas=None, ndigit=2):
    import xarray as xr

    if prds is None:
        prds = load_predictions(fmt='xarray')

    if ndigit not in (2, 4):
        raise ValueError('HLA ndigit must be 2 or 4')

    prds = filter_ndigit(prds, ndigit)

    # Calculate distance and linkage by hand for optimal ordering
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage
    from polo import optimal_leaf_ordering
    method = 'average'
    metric = 'euclidean'

    gs = {}
    for i, label in enumerate(prds.coords['probability'].values):
        if isinstance(prds, xr.DataArray):
            df = pd.DataFrame(
                    data=prds[:, :, i].values,
                    index=prds.indexes['hla'],
                    columns=prds.indexes['pid']
                    )

        Y_sample = pdist(df.values.T, metric=metric)
        Z_sample = linkage(Y_sample, method=method)
        Z_optimal_sample = optimal_leaf_ordering(Z_sample, Y_sample)
        Y_hla = pdist(df.values, metric=metric)
        Z_hla = linkage(Y_hla, method=method)
        Z_optimal_hla = optimal_leaf_ordering(Z_hla, Y_hla)

        pnames = sorted(
                set([tmp.split('_')[0] for tmp in df.columns]),
                key=lambda x: int(x[1:])
                )
        cols = sns.color_palette("husl", len(pnames))

        pnames_all = [tmp.split('_')[0] for tmp in df.columns]
        cols_all = [cols[pnames.index(pn)] for pn in pnames_all]

        g = sns.clustermap(
                df,
                row_linkage=Z_optimal_hla,
                col_linkage=Z_optimal_sample,
                col_colors=cols_all,
                vmin=0,
                vmax=1,
                )

        ax = g.ax_heatmap

        # There is a bug in the heatmap algorithm??
        ax.set_yticks(0.5 + np.arange(df.shape[0]))
        ax.set_xticks(0.5 + np.arange(df.shape[1]))
        ax.set_yticklabels(df.index[g.dendrogram_row.reordered_ind])
        ax.set_xticklabels(df.columns[g.dendrogram_col.reordered_ind])

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        ax.set_xlabel('patient-time')
        g.ax_col_colors.yaxis.set_label_position('right')
        g.ax_col_colors.set_ylabel('Patient', rotation=0, ha='left')

        g.fig.suptitle(label.capitalize())
        plt.subplots_adjust(left=0.02, top=0.95)

        # Add rectangles on the sample HLAs
        if label == 'posterior':
            from matplotlib.patches import Rectangle

            if hlas is None:
                hlas = load_HLA('data/HLA_types.csv')

            xlabels = [tk.get_text() for tk in ax.get_xticklabels()]
            ylabels = [tk.get_text() for tk in ax.get_yticklabels()]
            cols = {'A': 'red', 'B': 'green', 'C': 'blue'}

            for pname, tmp in hlas.items():
                for abc, tmp1 in tmp.items():
                    for s in tmp1:
                        sdig = abc+s[:2]
                        if ndigit == 4:
                            sdig += s[3:5]

                        j = None
                        for row in ylabels:
                            if row == sdig:
                                break
                        else:
                            continue
                        j = ylabels.index(row)

                        for i, samplename in enumerate(xlabels):
                            sample_pname = samplename.split('_')[0]
                            if pname != sample_pname:
                                continue

                            ax.add_patch(Rectangle(
                                (i, j), 1, 1,
                                facecolor='none', edgecolor=cols[abc], lw=1,
                                ))

        gs[label] = g

    return {'gs': gs, 'predictions': predictions}



# Script
if __name__ == '__main__':


    pa = argparse.ArgumentParser(description='Process some integers.')
    pa.add_argument('--trajectories', action='store_true',
                    help='Plot probability for trajectories')
    pa.add_argument('--heatmap', action='store_true',
                    help='Plot probability heatmap')

    args = pa.parse_args()

    # HLA types
    patients = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']
    HLAs = load_HLA('data/HLA_types.csv')

    if args.heatmap:
        predictions = load_predictions(fmt='xarray')
        d = plot_heatmap(prds=predictions)

        plt.ion()
        plt.show()

    if args.trajectories:
        # Prediction
        predictions = load_predictions(fmt='nested_dict')

        # trajectories of actual HLA types
        traj = {}
        for p in patients:
            traj[p] = {}
            for x in "ABC":
                for allele in HLAs[p][x]:
                    for suffix in ['']: #, '_ST']:
                        two_digit = x+allele.split(':')[0] + suffix
                        prob = []
                        for pat, day in predictions:
                            if p==pat:
                                if two_digit in predictions[(pat, day)]:
                                    prob.append([day]+list(predictions[(pat, day)][two_digit]))

                        if len(prob):
                            traj[p][two_digit] = np.array(sorted(prob, key=lambda x:x[0]))

        for p in patients:
            plt.figure()
            plt.title('%s_adaptation'%p)
            for allele in sorted(traj[p].keys()):
                plt.plot(traj[p][allele][:,0], traj[p][allele][:,1], label=allele)
            plt.legend(loc=2)
            plt.xlabel('time since infection')
            plt.ylabel('posterior')
            plt.savefig('figures/%s_adaptation.pdf'%p)

        # trajectories of top HLA types in first sample
        traj = {}
        initial_preds = {}
        for p in patients:
            initial_preds[p]={}
            first = sorted([day for pat, day in predictions if pat==p])[0]
            traj[p] = {}
            for x in "ABC":
                top_pred = sorted([(allele, posterior) for allele, (posterior, prior)
                                  in predictions[(p, first)].iteritems()
                                  if allele[0]==x], key=lambda x:x[1])[-1][0]
                allele=top_pred
                prob = []
                for pat, day in predictions:
                    if p==pat:
                        if allele in predictions[(pat, day)]:
                            prob.append([day]+list(predictions[(pat, day)][allele]))

                if len(prob):
                    initial_preds[p][allele] = np.array(sorted(prob, key=lambda x:x[0]))

        for p in patients:
            plt.figure()
            plt.title('%s_reversion'%p)
            for allele in sorted(initial_preds[p].keys()):
                plt.plot(initial_preds[p][allele][:,0], initial_preds[p][allele][:,1], label=allele)
            plt.legend(loc=2)
            plt.xlabel('time since infection')
            plt.ylabel('posterior')
            plt.savefig('figures/%s_reversion.pdf'%p)

        plt.ion()
        plt.show()
