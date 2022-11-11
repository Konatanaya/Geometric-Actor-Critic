import networkx as nx
import numpy as np
import utils
import multiprocessing as mp
import math
import torch
import time

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

    # def reset_action(self):
    #     self.action_matrix = np.argmax(self.preference_matrix, axis=1)
    #     self.incentive_matrix = np.zeros(len(self.user_id))
    #     self.feature_matrix = np.hstack((self.incentive_matrix.reshape(-1, 1), np.eye(self.num_of_actions)[self.action_matrix]))
    #     self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
    #     active_rate = np.sum(self.action_matrix == 0) / len(self.user_id)
    #
    #     return self.feature_matrix, active_rate, np.sum(self.action_matrix == 0)

    def reset(self):
        self.action_matrix = torch.argmax(self.preference_matrix, axis=1)
        self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
        incentive = torch.zeros((self.num_of_user, 1)).to(device)
        self.feature_matrix = torch.cat([incentive, self.action_one_hot_matrix], dim=1).flatten()
        # self.feature_matrix = self.action_one_hot_matrix.flatten()
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
        rewards = self.reward(incentives)+ rate - self.last_active_rate
        self.last_active_rate = rate

        return (self.feature_matrix,
                rewards.cpu().data.numpy().flatten(),
                self.last_active_rate.cpu().data.numpy().flatten(),
                active_num.cpu().data.numpy().flatten(),
                remaining_budget)

    def reward(self, incentives):
        return torch.sum((self.action_one_hot_matrix[:,0]*2-1)*(1 + (self.user_out_degree - self.user_in_degree)/self.num_of_user) + self.action_one_hot_matrix[:,0] * (1-incentives/self.budget))

    def sample_action(self):
        return torch.clip(torch.normal(0, 1, size=(1, self.num_of_user)), -1, 1).flatten().to(device)


# class Simulation(object):
#     def __init__(self, dataset, num_of_actions, budget):
#         self.G = utils.import_users_data_from_csv(dataset)
#         self.num_of_actions = num_of_actions
#         self.budget = budget
#
#         self.user_id = list(self.G.nodes())
#
#         self.preference_matrix = np.array([self.G.nodes[user]['Preference'] for user in self.user_id])
#         self.weight_dict = {user: np.array([[self.G[in_neighbor_id][user]['Weight'] for in_neighbor_id in self.G.predecessors(user)]]) for user in self.user_id}
#         self.action_matrix = None
#         self.feature_matrix = None
#         self.action_one_hot_matrix = None
#         self.weight_matrix = np.zeros((len(self.user_id), len(self.user_id)))
#
#         self.incentive_matrix = np.zeros(len(self.user_id))
#         self.user_index2id = {i: user for i, user in enumerate(self.user_id)}
#         self.user_id2index = {user: i for i, user in enumerate(self.user_id)}
#
#         # for user in self.user_id:
#         #     for in_neighbor_id in self.G.predecessors(user):
#         #         self.weight_matrix[self.user_id2index[user]][self.user_id2index[in_neighbor_id ]] = self.G[in_neighbor_id][user]['Weight']
#
#         self.adjacency_matrix_out = np.array(nx.adjacency_matrix(self.G).todense())
#         self.weight_matrix = torch.FloatTensor(nx.adjacency_matrix(self.G, weight="Weight").todense().transpose()).to(device)
#
#         self.adjacency_matrix_in = torch.FloatTensor(self.adjacency_matrix_out.transpose()).to(device)
#         self.torch_preference = torch.FloatTensor(self.preference_matrix).to(device)
#
#         self.preference_matrix_gpu = torch.FloatTensor(self.preference_matrix).to(device)
#         self.worker_num = mp.cpu_count()
#         self.index_range = [math.floor(len(self.user_id) / self.worker_num) * i for i in range(self.worker_num)] + [len(self.user_id)]
#         self.workers = []
#
#     def reset_action(self):
#         self.action_matrix = np.argmax(self.preference_matrix, axis=1)
#         self.incentive_matrix = np.zeros(len(self.user_id))
#         self.feature_matrix = np.hstack((self.incentive_matrix.reshape(-1, 1), np.eye(self.num_of_actions)[self.action_matrix]))
#         self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
#         active_rate = np.sum(self.action_matrix == 0) / len(self.user_id)
#
#         return self.feature_matrix, active_rate, np.sum(self.action_matrix == 0)
#
#     def reset_action_gpu(self):
#         self.action_matrix = torch.argmax(self.preference_matrix_gpu, axis=1)
#         # self.incentive_matrix = torch.zeros(len(self.user_id)).to(device)
#         # self.feature_matrix = np.hstack((self.incentive_matrix.reshape(-1, 1), np.eye(self.num_of_actions)[self.action_matrix]))
#         self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix].to(device)
#         # active_rate = np.sum(self.action_matrix == 0) / len(self.user_id)
#
#         # return self.feature_matrix, active_rate, np.sum(self.action_matrix == 0)
#
#     def one_hot_encoder(self, action):
#         targets = np.array([action])
#         one_hot = np.eye(self.num_of_actions)[targets].reshape(-1, self.num_of_actions)
#         return one_hot
#
#     def vectorized_one_time_step(self, inc_action):
#         influence = torch.matmul(self.weight_matrix, self.action_one_hot_matrix)
#         utility_matrix = self.torch_preference + influence
#         print(utility_matrix[0])
#         utility_dif = utility_matrix[:, 0] - torch.max(utility_matrix, 1)[0]
#
#         # print(utility_matrix[1])
#         # print(torch.max(utility_matrix, 1)[0])
#         incentives = (inc_action + 1) / 2
#         remaining_budget = torch.FloatTensor([self.budget]).to(device)
#         spent_incentives = torch.where(incentives + utility_dif >= 0, incentives, 0)
#
#         cum_incentives = torch.cumsum(spent_incentives, dim=0)
#         spent_incentives_ = torch.where(cum_incentives <= remaining_budget, spent_incentives, 0)
#         spent_incentives_index = torch.where(cum_incentives <= remaining_budget, 0, 1)
#         # print(spent_incentives_.nonzero())
#         remaining_budget -= torch.sum(spent_incentives_)
#         # print(remaining_budget)
#         if remaining_budget > 0:
#             final_spent_incentives = torch.where((spent_incentives_index == 1) & (utility_dif + remaining_budget >= 0), 1, 0)
#             # print(final_spent_incentives.nonzero())
#             spent_incentives_[final_spent_incentives.nonzero()[0]] = remaining_budget
#
#         # print(spent_incentives_.nonzero())
#
#         utility_matrix[:,0] += spent_incentives_
#
#         self.action_matrix = torch.argmax(utility_matrix, dim=1)
#         self.action_one_hot_matrix = torch.eye(self.num_of_actions)[self.action_matrix]
#
#     def one_time_step(self, inc_action):
#         influence_matrix = []
#         # self.create_workers()
#         for i in range(self.worker_num):
#             worker_range = self.index_range[i: i + 2] if i < self.worker_num - 1 else self.index_range[i:]
#             self.workers[i].inQ.put((self.user_id[worker_range[0]:worker_range[1]], np.eye(self.num_of_actions)[self.action_matrix]))
#         for w in self.workers:
#             influence = w.outQ.get()
#             influence_matrix += influence
#
#         utility_matrix = self.preference_matrix + np.array(influence_matrix)
#         incentives = (inc_action + 1) / 2
#         # print(inc_action, incentives)
#         remaining_budget = self.budget
#         for index, (utility, incentive) in enumerate(zip(utility_matrix, incentives)):
#             # print(utility_matrix[0], utility, incentive)
#             incentive = incentive if incentive <= remaining_budget else remaining_budget
#             padding = np.zeros(self.num_of_actions - 1)
#             if incentive < 0:
#                 print("!!!!")
#             incentive = np.array([incentive])
#             # print(incentive)
#             incentive = np.hstack((incentive, padding))
#
#             if np.argmax(utility + incentive) == 0:
#                 utility_matrix[index] += incentive
#                 remaining_budget -= incentive[0]
#                 # print(index, incentive[0], remaining_budget)
#                 if remaining_budget <= 0:
#                     remaining_budget = 0
#                     break
#         # print(incentives)
#         self.action_matrix = np.argmax(utility_matrix, axis=1)
#         # print(self.action_matrix[0], utility_matrix[0])
#         self.incentive_matrix = incentives
#         self.feature_matrix = np.hstack((self.incentive_matrix.reshape(-1, 1), np.eye(self.num_of_actions)[self.action_matrix]))
#         active_rate = np.sum(self.action_matrix == 0) / len(self.user_id)
#         rewards = np.mean(np.array([self.reward_function(user, action, incentive) for user, action, incentive in zip(self.user_id, self.action_matrix, self.incentive_matrix)])) + active_rate
#         # print(rewards)
#         # done = np.array([1 if time_step == total_steps else 0 for _ in range(len(self.user_id))]).reshape(-1, 1)
#         return self.feature_matrix, rewards, active_rate, np.sum(self.action_matrix == 0), self.incentive_matrix, remaining_budget
#
#     def reward_function(self, user, action, incentive):
#         # print(user, action, incentive)
#         if action == 0:
#             reward = (1 + (self.G.out_degree(user) - self.G.in_degree(user)) / len(self.user_id) + 1 - incentive)
#         else:
#             reward = -(1 + (self.G.out_degree(user) - self.G.in_degree(user)) / len(self.user_id))
#         return reward
#
#     def sample_action(self):
#         return np.clip(np.random.normal(0, 1, len(self.G.nodes())), -1, 1)
#
#     def create_workers(self, ):
#         for i in range(self.worker_num):
#             self.workers.append(Worker(mp.Queue(), mp.Queue(), self.G, self.weight_dict, self.user_index2id, self.user_id2index, self.num_of_actions))
#             self.workers[i].start()
#
#     def terminate_workers(self, ):
#         for w in self.workers:
#             w.terminate()
#         self.workers = []


class Worker(mp.Process):
    def __init__(self, inQ, outQ, G, weight_dict, index2id, id2index, num_of_actions):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.G = G
        self.weight_dict = weight_dict
        self.index2id = index2id
        self.id2index = id2index
        self.num_of_actions = num_of_actions

    def one_hot_encoder(self, action, num_of_actions):
        targets = np.array([action])
        one_hot = np.eye(num_of_actions)[targets].reshape(-1)
        return one_hot

    def reward_function(self, user_id, action, incentive, total_budget):
        if action == 0:
            reward = (1 + (self.G.out_degree(user_id) - self.G.in_degree(user_id)) / len(self.G.nodes()) + 1 - incentive / total_budget)
        else:
            reward = -(1 + (self.G.out_degree(user_id) - self.G.in_degree(user_id)) / len(self.G.nodes()))
        return reward

    def run(self):
        while True:
            data_block = self.inQ.get()
            index_range = data_block[0]
            action_matrix = data_block[1]
            influence = [np.sum(np.array([action_matrix[self.id2index[in_neighbor_id]] for in_neighbor_id in self.G.predecessors(self.index2id[user_index])]) * self.weight_dict[self.index2id[user_index]].T, axis=0) if len(
                list(self.G.predecessors(self.index2id[user_index]))) != 0 else np.zeros(self.num_of_actions) for user_index in index_range]
            self.outQ.put(influence)


if __name__ == '__main__':
    a = torch.FloatTensor([1, 2, 3])
    print(a[4 % 3])
    # env = Simulation("twitter", 4, 20)
    # # print(len(env.user_id))
    # torch.manual_seed(123)
    # np.random.seed(123)
    #
    # temp1 = []
    # temp2 = []
    # incentive = np.random.normal(0,1,len(env.user_id)).clip(-1,1)
    # env.create_workers()
    # start1 = time.time()
    # for ep in range(100):
    #     start2 = time.time()
    #     env.reset_action()
    #     for t in range(50):
    #         env.one_time_step(incentive)
    #         # env.vectorized_one_time_step(torch.zeros(len(env.user_id)).to(device))
    #         print(env.action_matrix)
    #         temp1.append(env.action_matrix)
    #     print(f" Ep {ep}: {time.time() - start2}s")
    # print(f"Total: {(time.time() - start1)}s")
    # env.terminate_workers()
    #
    # start1 = time.time()
    # for ep in range(100):
    #     start2 = time.time()
    #     env.reset_action()
    #     for t in range(50):
    #         # env.one_time_step(np.zeros(len(env.user_id)) - 1)
    #         env.vectorized_one_time_step(torch.FloatTensor(incentive).to(device))
    #         print(env.action_matrix)
    #         temp2.append(env.action_matrix.data.numpy())
    #     print(f" Ep {ep}: {time.time() - start2}s")
    # print(f"Total: {(time.time() - start1)}s")

    # for a,b in zip(temp1,temp2):
    #     print((a==b).all())
