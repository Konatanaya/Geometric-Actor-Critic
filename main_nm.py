import numpy as np
import networkx as nx
import utils
import random
from normal_approaches import *
import pyhocon
import argparse


class Simulation(object):
    def __init__(self, num_of_actions):
        self.num_of_actions = num_of_actions

    def reset_action(self, G):
        preference = np.array([tup[1]['Preference'] for tup in G.nodes(data=True)])
        actions = np.argmax(preference, axis=1)
        action_dict = {ID: [act] for ID, act in zip(G.nodes(), actions)}
        nx.set_node_attributes(G, action_dict, 'Action')

        incentives_dict = {ID: 0. for ID in G.nodes()}
        nx.set_node_attributes(G, incentives_dict, 'Incentive')

        features_dict = {ID: [np.hstack((np.array([incentives_dict[ID]]), self.one_hot_encoder(action_dict[ID])))] for ID in G.nodes()}
        nx.set_node_attributes(G, features_dict, 'Feature')

        active_count = np.sum(actions == 0)

        return active_count

    def one_hot_encoder(self, action):
        targets = np.array([action])
        one_hot = np.eye(self.num_of_actions)[targets].reshape(-1)
        return one_hot

    def one_step(self, G, incentives):
        preference = np.array([tup[1]['Preference'] for tup in G.nodes(data=True)])

        # incentives = np.array([[0.] for _ in G.nodes()])
        incentives_dict = {ID: inc[0] for ID, inc in zip(G.nodes(), incentives)}
        padding = np.zeros([G.number_of_nodes(), len(preference[0]) - 1])
        incentives = np.hstack((incentives, padding))

        influence = {ID: [0. for _ in preference[0]] for ID in G.nodes()}
        influence = {ID: self.update_influence_list(G, ID, v) for ID, v in influence.items()}
        influence = np.array([v for ID, v in influence.items()])

        utility = preference + incentives + influence
        actions = np.argmax(utility, axis=1)
        action_dict = {ID: act for ID, act in zip(G.nodes(), actions)}

        features_dict = {ID: G.nodes[ID]['Feature'] + [np.hstack((np.array([incentives_dict[ID]]), self.one_hot_encoder(action_dict[ID])))] for ID in G.nodes()}

        nx.set_node_attributes(G, features_dict, 'Feature')
        nx.set_node_attributes(G, action_dict, 'Action')
        nx.set_node_attributes(G, incentives_dict, 'Incentive')

        active_count = np.sum(actions == 0)
        # print(G.nodes(data=True))
        return active_count

    def one_step_(self, G, ID, incentive, step):
        preference = np.array(G.nodes[ID]['Preference'])

        padding = np.zeros(len(preference) - 1)
        incentive = np.array([incentive])
        incentive = np.hstack((incentive, padding))

        influence = [0. for _ in preference]
        influence = self.update_influence_list(G, ID, influence, step)

        utility = preference + incentive + influence
        action = np.argmax(utility)

        G.nodes[ID]['Action'].append(action)
        G.nodes[ID]['Incentive'] = incentive[0]
        f = np.hstack((np.array(incentive[0]), self.one_hot_encoder(action)))
        G.nodes[ID]['Feature'].append(f)

        return action

    def update_influence_list(self, G, node, v, step):
        for n in G.predecessors(node):
            act = G.nodes[n]['Action'][step-1]
            v[act] += G[n][node]['Weight']
        return v

def init_parser():
    parser = argparse.ArgumentParser(description="Hyper Parameters")
    parser.add_argument('--dataset', type=str, default='dolphins')
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--time_steps', type=int, default=150)
    parser.add_argument('--budget', type=int, default=3)

    args = parser.parse_args()
    parser.print_help()
    return args


def simulate(args, G, method):
    random.seed(args.seed)
    np.random.seed(args.seed)
    simulation = Simulation(4)

    total_num = []
    ID_list = list(G.nodes())
    active_num = simulation.reset_action(G)
    total_num.append(active_num)
    for time in range(1, args.time_steps + 1):
        remaining_budget = args.budget

        # random.shuffle(ID_list)
        if isinstance(method, DGIA):
            ID_list = method.sort_users(ID_list)
        active_num = 0
        for ID in ID_list:
            if isinstance(method, Uniform):
                incentive = method.allocate_incentives(ID, args.budget)
            else:
                incentive = method.allocate_incentives(ID, remaining_budget)

            action = simulation.one_step_(G, ID, incentive, time)
            if action == 0:
                remaining_budget -= incentive
                remaining_budget = 0 if remaining_budget <= 0 else remaining_budget
                active_num += 1
            if isinstance(method, DGIA):
                method.update_incentive_degree(ID, action)
                method.estimate_influence(ID, action, time - 1)
            if isinstance(method, DBPUCB):
                method.update_arm(ID, 1 if action == 0 else 0)
            if isinstance(method, kmab):
                method.update_arm(ID, 1 if action == 0 else 0)
        if isinstance(method, DGIA):
            method.calculate_influence_degree()
            method.status_last_time = active_num / len(ID_list)
        total_num.append(active_num)
        print(time, active_num, remaining_budget)
    np.save(f"./Results/{args.dataset}/performance/{method.name}_{args.budget}", total_num)
    utils.save_results(args.dataset, total_num, method.__class__.__name__)


if __name__ == '__main__':
    args = init_parser()
    G = utils.import_users_data_from_csv(args.dataset)
    approaches = [NoIn(G), Uniform(G), DGIA(G), DBPUCB(G), kmab(G)]
    for model in approaches[:]:
        simulate(args, G, model)
