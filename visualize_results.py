#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    degree = 3
    size = int(sys.argv[1])
    n_instance = 10
    n_realizations = 30

    # read and combine data
    dfs=[]
    for i in range(n_instance):
        dfs.append(pd.read_csv('payoffs/payoffs_size%d_i%d.csv' % (size, i)))
    df = pd.concat(dfs)  # columns: ['c', 'L', 'instance', 'realization', 'payoff']

    # number of payoffs for each (c,L) pair
    n_obs = n_instance * n_realizations

    ### relative payoff plot ### 
    # compute relative payoff
    df['relatie payoff'] = df.apply(
        lambda x: min(1,x['payoff']/df[(df['c']==x['c']) & 
                                          (df['instance']==x['instance']) & 
                                          (df['realization']==x['realization'])]\
        .sort_values('L', ascending=False)['payoff'].iloc[0]), axis=1)

    ### log(opt. gap) plot ### 
    # compute log(opt. gap)
    df['log(relative gap)'] = np.log10(1 - df['relatie payoff'] + 1e-5)

    # set up figure & colors
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    plt.subplots_adjust(wspace=.25)
    cmap = plt.get_cmap("tab10")
    markers = ['o', 'v', 's', 'P', 'D']

    # aggregate data -- computing mean and standard error
    df_agg = df.groupby(['c', 'L']).agg({'relatie payoff': ['mean', 'std']}).reset_index()
    df_agg[('relatie payoff', 'ste')] = df_agg[('relatie payoff', 'std')]/np.sqrt(n_obs)
    df_agg.sort_values(['c', 'L'], ascending=[True, True], inplace=True)

    # one line (with errors) for each c
    for j, c in enumerate(np.unique(df['c'])):
        xs = df_agg[df_agg['c']==c]['L']
        ys = df_agg[df_agg['c']==c][('relatie payoff', 'mean')]
        errs = df_agg[df_agg['c']==c][('relatie payoff', 'ste')]*2
        ax[0].errorbar(xs, ys, yerr=errs, marker=markers[j], markersize=8, 
                       color=cmap(j), label='c=%.2f'%(c), alpha=.7, ls='none')

    # figure labels and title
    ax[0].set_ylabel('relative payoff')
    ax[0].set_xlabel('locality parameter')
    ax[0].set_title('size=%d' % (size))

    # aggregate data -- computing mean and standard error
    df_agg = df.groupby(['c', 'L']).agg({'log(relative gap)': ['mean', 'std']}).reset_index()
    df_agg[('log(relative gap)', 'ste')] = df_agg[('log(relative gap)', 'std')]/np.sqrt(n_obs)

    # one line (with errors) for each c
    for j, c in enumerate(np.unique(df['c'])):
        xs = df_agg[df_agg['c']==c]['L']
        ys = df_agg[df_agg['c']==c][('log(relative gap)', 'mean')]
        errs = df_agg[df_agg['c']==c][('log(relative gap)', 'ste')]*2
        ax[1].errorbar(xs, ys, yerr=errs, marker=markers[j],  markersize=8,
                       color=cmap(j), label='c=%.2f'%(c), alpha=.7, ls='none')

    # figure labels and title
    ax[1].set_ylabel('1-relative payoff (log scale)')
    ax[1].set_xlabel('locality parameter')
    ax[1].set_title('size=%d' % (size))

    # set up yticks and yticklabels
    ax[1].set_yticks([-5,-4,-3,-2,-1,0])
    ax[1].set_yticklabels([0, ' ', 0.001, 0.01, 0.1, 1], fontsize=8)

    # break the y-axis
    [x_lft, x_rgt] = ax[1].get_xlim()
    ax[1].plot(x_lft, -4, marker='s', color='w', markersize=20, zorder=10, clip_on=False)
    d = .05  # how big to make the diagonal lines in axes coordinates
    offset = .22
    ax[1].plot((x_lft-d*4, x_lft+d*4), (-4+offset-d, -4+offset+d), 
               color='k', zorder=10, clip_on=False)        # top-left diagonal
    ax[1].plot((x_lft-d*4, x_lft+d*4), (-4-offset-d, -4-offset+d), 
               color='k', zorder=10, clip_on=False)        # top-left diagonal
    ax[1].set_xlim([x_lft, x_rgt])

    # legend
    ax[1].legend(loc='best', fontsize=10)

    # save figure
    plt.savefig('results/figure_size%d_degree%d_payoff_vs_locality_synthetic.png' % (size, degree), 
                bbox_inches='tight')
    plt.close()
