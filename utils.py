import os
import numpy as np
import networkx as nx
import pandas as pd

item_num = 4
path = './Dataset/'


def read_dataset(filename):
    dataset = path + filename + '/' + filename + '.txt'
    network = {'Source': [], 'Target': [], 'Weight': []}
    users_preference = dict()
    remaining_weights = dict()
    user_id_map = dict()
    with open(file=dataset, mode='r') as f:
        for line in f.readlines():
            data = line.strip().split()
            source, target = int(data[0]), int(data[1])

            # network
            if target not in remaining_weights:
                remaining_weights[target] = 1.
            # if source not in network[target]:
            weight = np.random.uniform(0, 1) * remaining_weights[target]
            network['Source'].append(source)
            network['Target'].append(target)
            network['Weight'].append(weight)
            remaining_weights[target] -= weight

            # preference
            if source not in users_preference:
                users_preference[source] = [np.random.uniform(0, 1) for t in range(item_num)]
                users_preference[source] = [source] + users_preference[source]
            if target not in users_preference:
                users_preference[target] = [np.random.uniform(0, 1) for t in range(item_num)]
                users_preference[target] = [target] + users_preference[target]

    df_preference = pd.DataFrame.from_dict(users_preference, orient='index')
    df_preference.rename(columns={0: 'ID'}, inplace=True)
    df_preference.to_csv(path + filename + '/' + filename + '_preference.csv', index=False)

    df_network = pd.DataFrame.from_dict(network)
    df_network.to_csv(path + filename + '/' + filename + '_network.csv', index=False)
    print(len(users_preference), sorted(users_preference.keys()))


def import_users_data_from_csv(filename):
    preference_path = path + filename + '/' + filename + '_preference.csv'
    network_path = path + filename + '/' + filename + '_network.csv'
    network = pd.read_csv(network_path)
    users_preference = pd.read_csv(preference_path)
    G = nx.from_pandas_edgelist(network, 'Source', 'Target', ['Weight'], create_using=nx.DiGraph())

    preference_dict = users_preference.to_dict(orient='index')
    ID_dict = {v['ID']: v['ID'] for k, v in preference_dict.items()}
    preference_dict = {v['ID']: [p for _, p in v.items()][1:] for k, v in preference_dict.items()}

    nx.set_node_attributes(G, ID_dict, 'ID')
    nx.set_node_attributes(G, preference_dict, 'Preference')

    # adj_list = {node_id: set(G.successors(node_id)) for node_id in G.nodes()}
    return G


def save_results(dataset, results, method):
    file_path = '../Results/' + dataset
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + '/' + method + '.txt', mode='w') as f:
        for data in results:
            f.write(str(data) + '\n')


def process_log(dataset):
    path = '../Results/train/'
    results = []
    with open(path + dataset + '.out', mode='r') as f:
        for index, line in enumerate(f.readlines()):
            data = line.strip().split(' ')
            if (index != 0 and data[1] == '0') or data[0] != 'Epi:':
                continue
            else:
                results.append(data[6])

    with open(path + dataset + '.txt', mode='w') as f:
        for data in results:
            f.write(data + '\n')


if __name__ == '__main__':
    read_dataset('food')
    # load_nodes_and_edges('twitter')
    # import_users_data_from_csv('twitter')
    # process_log('wiki')
