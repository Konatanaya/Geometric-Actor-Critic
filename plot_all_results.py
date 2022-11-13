import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

TIME_STEPS = 2
DATASET = ['twitter', 'email', 'dolphins']
DATASET_INDEX = 1
BUDGET = 200


def generate_dict():
    dataset = {
        'wiki': {
            'num_of_users': 889,
            'y_max': 600,
            'y_min': 150,
            'y_interval': 50
        },
        'twitter': {
            'num_of_users': 236,
            'y_max': 200,
            'y_min': 25,
            'y_interval': 25
        },
        'dolphins': {
            'num_of_users': 62,
            'y_max': 60,
            'y_min': 10,
            'y_interval': 10
        },
    }
    graph = {
        'NoIn': {
            'name': 'No Incentive',
            'marker': 'o',
            'color': 'orange'
        },
        'Uniform': {
            'name': 'Uniform',
            'marker': 's',
            'color': 'royalblue'
        },
        'DGIA': {
            'name': 'DGIA-IPE',
            'marker': 's',
            'color': 'darkgreen'
        },
        'DBPUCB': {
            'name': 'DBP-UCB',
            'marker': '>',
            'color': 'grey'
        },
        'kmab':{
            'name':'K-MAB',
            'marker': '^',
            'color': 'red'
        },
        'GHRLG': {
            'name': 'GAC',
            'marker': 'x',
            'color': 'crimson'
        }
    }
    return graph, dataset


def draw_train_graph():
    path = '../Results/train/'
    files = os.listdir(path)
    results = dict()
    for file in files:
        if file[-4:] == '.txt':
            with open(file=path + file, mode='r') as f:
                result = []
                for line in f.readlines():
                    result.append(float(line.strip()))
                results[file[:-4]] = list(np.array(result[:]))
                length = len(result)
    print(results)

    l = ['dolphins', 'twitter', 'wiki']
    fig = plt.figure()
    for key in l:
        if key not in results.keys():
            continue
        value = results[key]
        plt.plot(value, ms=8)
        # legend.append(key)
    plt.legend(l, fontsize=12)
    plt.savefig('../plot/training.pdf', bbox_inches='tight')
    plt.show()


def draw_exploration(dataset, dataset_config, span=1):
    path = '../Results/exploration/' + dataset + '/'
    files = os.listdir(path)
    results_num = dict()
    results_r = dict()
    for file in files:
        if file[-5:-4] == 'r':
            with open(file=path + file, mode='r') as f:
                result = []
                for line in f.readlines():
                    result.append(float(line.strip()))
                results_r[file[:5]] = list(np.array(result[:]))
                length = len(result)
        else:
            with open(file=path + file, mode='r') as f:
                result = []
                for line in f.readlines():
                    result.append(float(line.strip()))
                results_num[file[:5]] = list(np.array(result[:]))
                length = len(result)

    l = ['norma', 'ratio', 'stand']
    m = ['s', 'o', '>']
    # c = ['b', 'r', 'g']
    fig = plt.figure()

    x = np.arange(0, length, span)
    for marker, key in zip(m, l):
        if key not in results_r.keys():
            continue
        value = results_r[key]
        plt.scatter(x, value[::span], alpha=0.5, s=0.5, )
        # plt.plot(x, value[:], ms=8, marker=marker)
        # legend.append(key)
    plt.legend([r'$\mathcal{N}(0,0.2)$', r'$\mathcal{N}(-\omega,1)$', r'$\mathcal{N}(0,1)$'], fontsize=12, markerscale=8, loc='lower right', framealpha=1, fancybox=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Cumulative step rewards', fontsize=12)
    plt.grid()
    plt.savefig('../plot/' + dataset + '_exp_r.eps', bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure()
    x = np.arange(0, 150 + 1, 5)
    for marker, key in zip(m, l):
        if key not in results_num.keys():
            continue
        value = results_num[key]
        # plt.scatter(x, value[::span], alpha=0.5, s=0.8, )
        plt.plot(x, value[::5], ms=6, marker=marker, mec='black')
        # legend.append(key)
    plt.legend([r'$\mathcal{N}(0,0.2)$', r'$\mathcal{N}(-\omega,1)$', r'$\mathcal{N}(0,1)$'], fontsize=12, loc='lower right', framealpha=1, fancybox=True)
    plt.xticks(np.arange(0, 151, 25), fontsize=12)
    plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)

    plt.xlabel('Time steps', fontsize=12)
    plt.ylabel('Number of users', fontsize=12)
    plt.grid()
    plt.savefig('../plot/' + dataset + '_exp_num.eps', bbox_inches='tight', dpi=144)
    plt.show()


def draw_variants(dataset, dataset_config, span=5):
    path = '../Results/variants/' + dataset + '/'
    files = os.listdir(path)
    results = dict()
    for file in files:
        if file[-4:] == '.txt':
            with open(file=path + file, mode='r') as f:
                result = []
                for line in f.readlines():
                    result.append(float(line.strip()))
                results[file[:-4]] = list(np.array(result[:]))
                length = len(result)

    l = ['GAC', 'GAC-IN', 'GAC-OUT']
    m = ['s', 'o', '>']
    c = ['orange', 'blue', 'red']
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    # plt.figure()
    # plt.style.use('ggplot')
    x = np.arange(0, length + 1, span)
    for marker, key, color in zip(m, l, c):
        if key not in results.keys():
            continue
        value = results[key]
        plt.plot(x, value[::span], ms=6, marker=marker, mec='black', color=color)
        # legend.append(key)
    plt.legend(l, fontsize=12, loc='lower right', framealpha=1, fancybox=True)

    plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] +1, dataset_config['y_interval']), fontsize=12)

    plt.xlabel('Time steps', fontsize=12)
    plt.ylabel('Number of users', fontsize=12)

    if dataset == 'twitter':
        plt.style.use('ggplot')
        axins = ax.inset_axes((0.5, 0.3, 0.4, 0.3))
        for marker, key, color in zip(m, l, c):
            if key not in results.keys():
                continue
            value = results[key]
            axins.plot(x, value[::span], ms=6, marker=marker, mec='black', color=color)

        zone_left = 1
        zone_right = 3
        x_ratio = 0.3
        y_ratio = 0.3
        xlim0 = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
        xlim1 = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio
        y = np.hstack((results['GAC'][::span][zone_left:zone_right], results['GAC-IN'][::span][zone_left:zone_right], results['GAC-OUT'][::span][zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.grid()
    plt.savefig('../plot/' + dataset + '_variants.eps', bbox_inches='tight')
    plt.show()


def draw_graph(dataset, dataset_config, graph, span=5):
    results = dict()
    files = os.listdir('../Results/' + dataset + '/')
    for file in files:
        with open(file='../Results/' + dataset + '/' + file) as f:
            result = []
            for line in f.readlines():
                result.append(float(line.strip()))
            results[file[:-4]] = list(np.array(result[:]))
            length = len(result)
    print(results)

    l = ['NoIn', 'Uniform', 'DGIA', 'DBPUCB', 'kmab', 'GHRLG']
    legend = []
    plt.figure()
    x = np.arange(0, length + 1, span)
    for key in l:
        if key not in results.keys():
            continue
        value = results[key]
        plt.plot(x, (value)[::span], ms=6, color=graph[key]['color'], marker=graph[key]['marker'], mec='black')
        legend.append(graph[key]['name'])
    plt.legend(legend, fontsize=12, loc='lower center', ncol=3, framealpha=1, fancybox=True)
    plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)

    plt.xlabel('Time steps', fontsize=12)
    plt.ylabel('Number of users', fontsize=12)
    plt.grid()
    plt.savefig('../plot/' + dataset + '.eps', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    # dataset = DATASET[DATASET_INDEX]
    plt.style.use('seaborn-dark')

    graph, datasets = generate_dict()

    for dataset, config in datasets.items():
        draw_graph(dataset, config, graph)
