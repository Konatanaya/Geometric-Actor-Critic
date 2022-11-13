import numpy as np
import math


class NoIn(object):
    def __init__(self, G):
        self.G = G
        self.name = "No Incentive"

    def allocate_incentives(self, user_id, budget):
        incentive = 0.
        return incentive


class Uniform(object):
    def __init__(self, G):
        self.G = G
        self.name = "Uniform"

    def allocate_incentives(self, user_id, budget):
        incentive = budget / len(self.G.nodes())
        # incentives = np.array([[incentive] for _ in self.G.nodes()])
        return incentive


class DBPUCB(object):
    def __init__(self, G, arm_num=100):
        self.G = G
        self.arm_num = arm_num
        self.price = [i * 0.01 for i in range(1, self.arm_num + 1)]
        self.times = [1 for _ in range(self.arm_num)]
        self.F = [0 for _ in range(self.arm_num)]
        self.count = 0

        self.result_dict = {}
        self.name = "DBP-UCB"

    def update_arm(self, user_id, result):
        index = self.result_dict[user_id]
        self.times[index] += 1
        self.F[index] = self.F[index] + (result - self.F[index]) / self.times[index]

    def allocate_incentives(self, user_id, remaining_budget):
        if remaining_budget <= 0:
            self.result_dict[user_id] = 0
            return 0.
        else:
            self.count += 1
            compare = remaining_budget / len(self.G.nodes())
            F_ = [f + math.sqrt(2 * math.log(self.count) / self.times[index]) for index, f in enumerate(self.F)]
            temp = [min(f, compare / p) for p, f in zip(self.price, F_)]
            # temp_ = [i if self.price[index] < remaining_budget else remaining_budget for index, i in enumerate(temp)]
            index = np.argmax(temp)
            incentive = self.price[index]
            incentive = remaining_budget if incentive >= remaining_budget else incentive
            self.result_dict[user_id] = index
            # print(user_id)
        return incentive


class kmab(object):
    def __init__(self, G, arm_num=100):
        self.G = G
        self.arm_num = arm_num
        self.price = [i * 0.01 for i in range(1, self.arm_num + 1)]
        self.P = [0.5 for i in range(arm_num)]
        self.increment = 0.01
        self.min_incentive = 0
        self.max_incentive = 1
        self.budget = 0
        self.times = [1 for _ in range(self.arm_num)]
        self.count = 0

        self.price_index = -1
        self.result_dict = {}
        self.name = "K-MAB"

    def update_arm(self, user_id, result):
        r = result
        temp_p = self.P[self.price_index]
        self.P[self.price_index] = temp_p + (r - temp_p) / (self.count + 1)
        self.count += 1


    def allocate_incentives(self, user_id, remaining_budget):
        if self.count == 0:
            self.budget = remaining_budget
        if remaining_budget <= 0:
            self.result_dict[user_id] = 0
            return 0.
        else:
            temp = [min(self.budget / self.price[i], self.P[i] * len(self.G.nodes())) for i in range(self.arm_num)]
            # print(self.budget, temp)
            self.price_index = np.argmax(temp)
            incentive = self.price[self.price_index]
        return incentive


class DGIA(object):
    def __init__(self, G):
        self.G = G
        self.omega = {user_id: self.G.nodes[user_id]['Preference'][0] / np.sum(self.G.nodes[user_id]['Preference']) for
                      user_id in self.G.nodes()}
        self.incentive_degree = {user_id: .5 for user_id in self.G.nodes()}
        self.influence_degree = {user_id: 0. for user_id in self.G.nodes()}
        self.influence = {k: {k_: 0. for k_ in self.G.nodes() if k_ != k} for k in self.G.nodes()}
        # self.influence_method = IPE(self.G.nodes())
        self.status_last_time = 0.
        self.beta = .1
        self.name = "DGIA-IPE"

    def update_incentive_degree(self, user_id, action):
        value = self.incentive_degree[user_id]
        omega = self.omega[user_id]
        if action == 0:
            self.incentive_degree[user_id] = value / (value + omega * (1 - value))
        else:
            self.incentive_degree[user_id] = value * .8

    def allocate_incentives(self, user_id, remaining_budget):
        if remaining_budget <= 0:
            return 0
        else:
            preference = self.G.nodes[user_id]['Preference']
            pre_dif = np.max(preference) - preference[0]
            idegree = self.influence_degree[user_id]
            incen_degree = self.incentive_degree[user_id]
            a = pre_dif ** self.status_last_time
            b = idegree ** self.status_last_time
            reward = (1. - incen_degree) * (a + b)
            reward = remaining_budget if reward > remaining_budget else reward
        return reward

    def sort_users(self, user_list):
        return sorted(user_list, key=lambda x: (self.incentive_degree[x] + self.influence_degree[x]), reverse=True)

    def estimate_influence(self, user_id, user_action, time):
        for user_id_ in self.G.nodes():
            if user_id == user_id_:
                continue
            actions_ = self.G.nodes[user_id_]['Action']
            x = self.influence[user_id_][user_id]
            if user_action == actions_[time]:
                x = x + (math.exp(x - 1) - self.beta) * self.beta
            else:
                x = x - (math.exp(-x) - self.beta) * self.beta
            if x > 1.0:
                x = 1.0
            elif x < 0.0:
                x = 0.0
            self.influence[user_id_][user_id] = x

    def calculate_influence_degree(self):
        denominator = len(self.G.nodes())
        self.influence_degree = {k: np.sum(list(v.values())) / denominator for k, v in self.influence.items()}
        return self.influence_degree
