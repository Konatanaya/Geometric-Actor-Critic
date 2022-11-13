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
        'No Incentive': {
            'name': 'No Incentive',
            'marker': 'o',
            'color': 'orange'
        },
        'Uniform': {
            'name': 'Uniform',
            'marker': 's',
            'color': 'royalblue'
        },
        'DGIA-IPE': {
            'name': 'DGIA-IPE',
            'marker': 's',
            'color': 'darkgreen'
        },
        'DBP-UCB': {
            'name': 'DBP-UCB',
            'marker': '>',
            'color': 'grey'
        },
        'K-MAB': {
            'name': 'K-MAB',
            'marker': '^',
            'color': 'red'
        },
        'GAC': {
            'name': 'GAC',
            'marker': 'x',
            'color': 'crimson'
        }
    }
    return graph, dataset


def plot_performance(args, graph, dataset_config, span=5):
    path = f"Results/{args.dataset}/performance"
    files = os.listdir(path)
    print(files)

    results = dict()
    legend = []
    for f in files:
        data = np.load(f"{path}/{f}")
        method = f.split('_')[0]
        legend.append(method)
        results[method] = data
        length = len(data)

    plt.figure()
    x = np.arange(0, length + 1, span)
    for key in legend:
        if key not in results.keys():
            continue
        value = results[key]
        plt.plot(x, (value)[::span], ms=6, color=graph[key]['color'], marker=graph[key]['marker'], mec='black')
        # legend.append(graph[key]['name'])
    print(legend)
    plt.legend(legend, fontsize=12, loc='lower center', ncol=3, framealpha=1, fancybox=True)
    plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)

    plt.xlabel('Time steps', fontsize=12)
    plt.ylabel('Number of users', fontsize=12)
    plt.grid()
    # plt.savefig('./' + args.dataset + '.eps', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    args = init_parser()
    print("---------------------------------------")
    print(f"Dataset: {args.dataset} | Budget: {args.budget}")
    print("---------------------------------------")

    plt.style.use('seaborn-dark')

    graph, dataset_config = generate_dict()
    plot_performance(args, graph, dataset_config[args.dataset])
