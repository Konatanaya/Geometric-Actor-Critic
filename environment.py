import networkx as nx
import numpy as np
import utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulation(object):
    def __init__(self, dataset, num_of_actions, budget):
        self.G = utils.import_users_data_from_csv(dataset)
        self.num_of_actions = num_of_actions
        self.budget = budget
        self.user_id = list(self.G.nodes())
        self.user_index2id = {i: user for i, user in enumerate(self.user_id)}
        self.user_id2index = {user: i for i, user in enumerate(self.user_id)}
        self.num_of_user = len(self.user_id)

        # self.gpu_num_of_user = torch.Tensor([self.num_of_user]).to(device)

        self.adjacency_matrix_out = torch.FloatTensor(np.array(nx.adjacency_matrix(self.G).todense()))
        self.adjacency_matrix_in = self.adjacency_matrix_out.transpose(1, 0)

        self.preference_matrix = torch.FloatTensor(np.array([self.G.nodes[user]['Preference'] for user in self.user_id])).to(device)
        self.weight_matrix = torch.FloatTensor(nx.adjacency_matrix(self.G, weight="Weight").todense().transpose()).to(device)

        self.user_in_degree = torch.sum(self.adjacency_matrix_in, dim=1).to(device)
        self.user_out_degree = torch.sum(self.adjacency_matrix_out, dim=1).to(device)

        self.action_matrix = None
        self.feature_matrix = None
        self.action_one_hot_matrix = None

        self.last_active_rate = 0

    def reset(self):
        self.action_matrix = torch.argmax(self.preference_matrix, axis=1)
        self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
        incentive = torch.zeros((self.num_of_user, 1)).to(device)
        self.feature_matrix = torch.cat([incentive, self.action_one_hot_matrix], dim=1).flatten()
        active_num = torch.sum(self.action_matrix == 0)
        self.last_active_rate = active_num / self.num_of_user

        return self.feature_matrix, self.last_active_rate.cpu().data.numpy().flatten(), active_num.cpu().data.numpy().flatten()

    def step(self, inc_action):
        influence = torch.matmul(self.weight_matrix, self.action_one_hot_matrix)
        utility_matrix = self.preference_matrix + influence
        utility_dif = utility_matrix[:, 0] - torch.max(utility_matrix, 1)[0]
        incentives = (inc_action + 1) / 2
        remaining_budget = torch.FloatTensor([self.budget]).to(device)

        spent_incentives = torch.where(incentives + utility_dif >= 0, incentives, 0)  # Find incentives would be accepted
        # Calculate remaining budget
        cum_incentives = torch.cumsum(spent_incentives, dim=0)
        spent_incentives_ = torch.where(cum_incentives <= remaining_budget, spent_incentives, 0)

        spent_incentives_index = torch.where(cum_incentives <= remaining_budget, 0, 1)
        remaining_budget -= torch.sum(spent_incentives_)
        if remaining_budget > 0:
            final_spent_incentives = torch.where((spent_incentives_index == 1) & (utility_dif + remaining_budget >= 0), 1, 0)
            if len(final_spent_incentives.nonzero())!=0:
                spent_incentives_[final_spent_incentives.nonzero()[0]] = remaining_budget
        # print(utility_matrix.shape, print(spent_incentives_.shape))
        utility_matrix[:, 0] += spent_incentives_

        self.action_matrix = torch.argmax(utility_matrix, dim=1)
        self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
        self.feature_matrix = torch.cat([spent_incentives_.view(-1,1), self.action_one_hot_matrix], dim=1).flatten()
        # self.feature_matrix = self.action_one_hot_matrix.flatten()
        active_num = torch.sum(self.action_matrix == 0)
        rate = active_num / self.num_of_user
        rewards = self.reward(incentives)
        self.last_active_rate = rate

        return (self.feature_matrix,
                rewards.cpu().data.numpy().flatten(),
                self.last_active_rate.cpu().data.numpy().flatten(),
                active_num.cpu().data.numpy().flatten(),
                remaining_budget)

    def reward(self, incentives):
        return torch.mean((self.action_one_hot_matrix[:,0]*2-1)*(1 + (self.user_out_degree - self.user_in_degree)/self.num_of_user) + self.action_one_hot_matrix[:,0] * (1-incentives/self.budget))

    def sample_action(self):
        return torch.clip(torch.normal(0, 1, size=(1, self.num_of_user)), -1, 1).flatten().to(device)
