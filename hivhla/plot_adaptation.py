import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_predictions(fname=None, fmt='raw'):
    if fname is None:
        fname = 'data/from_Jonathan/All_hlaProb.txt'

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

    if fmt in ('pandas', 'xarray'):
        prds = pd.DataFrame(prds)
        prds_values = np.array([[[v[0], v[1]] for v in row] for row in df.values])
        prds = pd.Panel(
                data=prds_values,
                items=prds.index,
                major_axis=prds.columns,
                minor_axis=['posterior', 'prior'])

    if fmt == 'xarray':
        import xarray as xr
        prds = xr.DataArray(
            prds.values,
            coords={'hla': prds.items,
                    'sample': prds.major_axis,
                    'probability': prds.minor_axis},
            dims=['hla', 'sample', 'probability'])

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


def plot_all(prds=None):
    import xarray as xr

    if prds is None:
        prds = load_predictions(fmt='xarray')

    if isinstance(prds, xr.DataArray):
        post = pd.DataFrame(prds[:, :, 0].values, index=prds.indexes['hla'], columns=prds.indexes['sample'])
        prio = pd.DataFrame(prds[:, :, 1].values, index=prds.indexes['hla'], columns=prds.indexes['sample'])
    elif isinstance(prds, pd.Panel):
        post = prds.iloc[:, :, 0]
        prio = prds.iloc[:, :, 1]

    # Calculate distance and linkage by hand for optimal ordering
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage
    from polo import optimal_leaf_ordering
    method = 'average'
    metric = 'euclidean'

    gs = {}
    for label, df in [('Posterior', post), ('Prior', prio)]:
        Y_sample = pdist(df.values.T, metric=metric)
        Z_sample = linkage(Y_sample, method=method)
        Z_optimal_sample = optimal_leaf_ordering(Z_sample, Y_sample)
        Y_hla = pdist(df.values, metric=metric)
        Z_hla = linkage(Y_hla, method=method)
        Z_optimal_hla = optimal_leaf_ordering(Z_hla, Y_hla)

        pnames = sorted(post.columns.levels[0].tolist(), key=lambda x: int(x[1:]))
        cols = sns.color_palette("Set2", len(pnames))
        pnames_all = [tmp[0] for tmp in post.columns]
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
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        ax.set_xlabel('patient-time')
        g.fig.suptitle(label)
        plt.subplots_adjust(left=0.02, top=0.95)

        gs[label] = g

    return gs



# Script
if __name__ == '__main__':

    # Prediction
    predictions = load_predictions()

    # HLA types
    patients = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']
    HLAs = load_HLA('data/HLA_types.csv')

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

    plt.ion()
    plt.show()

    # FIXME
    import sys
    sys.exit()


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
