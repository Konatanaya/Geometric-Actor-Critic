import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Hyper Parameters")
    parser.add_argument('-d', '--dataset', type=str, default='twitter')
    parser.add_argument('-b', '--budget', type=int, default=20)
    parser.add_argument('-p', '--policy', type=str, default="TD3")
    parser.add_argument('-tp', '--training_plot', action='store_true', default=False)
    parser.add_argument('-pp', '--performance_plot', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    parser.print_help()
    return args


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
        'kmab': {
            'name': 'TD3',
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


def plot_training(args):
    path = f"Results/{args.dataset}/eval"
    files = os.listdir(path)
    print(files)
    plt.figure()
    legend = []
    for f in files:
        a = np.load(f"{path}/{f}")
        print(a)
        plt.plot(a[1:, 0], a[1:, 1], ms=8)
        legend.append(f)
    plt.legend(legend)
    plt.show()


def plot_performance(args):
    path = f"Results/{args.dataset}/performance"
    files = os.listdir(path)
    print(files)
    for f in files:
        a = np.load(f"{path}/{f}")
        print(a)


def draw_graph(dataset, dataset_config, graph, span=5):
    results = dict()
    path = f"Results/{args.dataset}/performance"
    files = os.listdir(path)
    for f in files:
        a = np.load(f"{path}/{f}")
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
    args = init_parser()
    print("---------------------------------------")
    print(f"Dataset: {args.dataset} | Budget: {args.budget}")
    print("---------------------------------------")

    plt.style.use('seaborn-dark')
    graph, datasets = generate_dict()

    if args.training_plot:
        plot_training(args)
    if args.performance_plot:
        plot_performance(args)
