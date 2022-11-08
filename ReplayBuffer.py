import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, max_size=int(1e5)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = torch.zeros((max_size, state_dim)).to(device)
#         self.action = torch.zeros((max_size, action_dim)).to(device)
#         self.reward = torch.zeros((max_size, 1)).to(device)
#         self.not_done = torch.zeros((max_size, 1)).to(device)
#
#     def add(self, state, action, reward, done):
#         # print(state.shape, action.shape, next_state.shape, reward, done)
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.reward[self.ptr] = torch.FloatTensor([reward])
#         self.not_done[self.ptr] = done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#         ind_ = (ind + 1) % self.size
#
#         return (
#             self.state[ind],
#             self.action[ind],
#             self.state[ind_],
#             self.reward[ind],
#             self.not_done[ind]
#         )

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, reward, done):
        # print(state.shape, action.shape, next_state.shape, reward, done)
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        ind_ = (ind + 1) % self.size

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.state[ind_]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, num_of_users, model, max_size=int(1e5)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         if model == "GAC":
#             self.state = np.zeros((max_size, num_of_users, state_dim))
#             self.action = np.zeros((max_size, num_of_users))
#             self.next_state = np.zeros((max_size, num_of_users, state_dim))
#             self.reward = np.zeros((max_size, 1))
#             self.not_done = np.zeros((max_size, 1))
#         elif model in ["TD3", "DDPG", "CLGAC"]:
#             self.state = np.zeros((max_size, state_dim))
#             self.action = np.zeros((max_size, action_dim))
#             self.next_state = np.zeros((max_size, state_dim))
#             self.reward = np.zeros((max_size, 1))
#             self.not_done = np.zeros((max_size, 1))
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, next_state, reward, done):
#         # print(state.shape, action.shape, next_state.shape, reward, done)
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def add_batch(self, state, action, next_state, reward, done):
#         for s, a, s_, r, d in zip(state, action, next_state, reward, done):
#             self.state[self.ptr] = s
#             self.action[self.ptr] = a
#             self.next_state[self.ptr] = s_
#             self.reward[self.ptr] = r
#             self.not_done[self.ptr] = 1. - d
#
#             self.ptr = (self.ptr + 1) % self.max_size
#             self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#
#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.reward[ind]).to(self.device),
#             torch.FloatTensor(self.not_done[ind]).to(self.device)
#         )


# class ReplayGoalBuffer(object):
#     def __init__(self, state_dim, action_dim, goal_dim, max_size=int(1e5)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.goal = np.zeros((max_size, goal_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.gamma = np.zeros((max_size,1))
#         self.not_done = np.zeros((max_size, 1))
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, goal, next_state, reward, gamma, done):
#         # print(state.shape, action.shape, next_state.shape, reward, done)
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.goal[self.ptr] = goal
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.gamma[self.ptr] = gamma
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#
#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.reward[ind]).to(self.device),
#             torch.FloatTensor(self.goal[ind]).to(self.device),
#             torch.FloatTensor(self.gamma[ind]).to(self.device),
#             torch.FloatTensor(self.not_done[ind]).to(self.device)
#         )

if __name__ == '__main__':
    a = torch.FloatTensor([[[1,1],
                            [2,2,]],
                           [[3,3],
                            [4,4]]])

    b = torch.FloatTensor([[1,2],
                           [3,1]])

    # a= torch.FloatTensor([[1,1],
    #                       [2,2]])
    print(torch.matmul(a,b))