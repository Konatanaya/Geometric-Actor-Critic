import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, adj):
        num_neighbors = adj.sum(1, keepdim=True)
        mean_adj = adj.div(num_neighbors)
        mean_adj = torch.where(torch.isnan(mean_adj), torch.full_like(mean_adj, 0), mean_adj)
        final_embedding = mean_adj.matmul(features)

        return final_embedding


class SAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim, base_model=None):
        super(SAGELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if base_model is not None:
            self.base_model = base_model

        self.aggregator = MeanAggregator()
        self.W = nn.Linear(self.input_dim * 2, self.output_dim)

    def forward(self, features, adj_matrix):
        aggregated_features = self.aggregator(features, adj_matrix)
        combined = torch.cat([features, aggregated_features], dim=-1)
        final_embeddings = torch.relu(self.W(combined))
        # final_embeddings = final_embeddings / torch.norm(final_embeddings, p=2, dim=-1, keepdim=True)
        return final_embeddings


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer=1):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layer

        self.sageLayers = nn.ModuleList()
        self.sageLayers.append(SAGELayer(self.input_dim, self.hidden_dim))
        for index in range(0, self.num_layers - 1):
            self.sageLayers.append(SAGELayer(self.hidden_dim, self.hidden_dim, base_model=self.sageLayers[index]))

    def forward(self, features, adj_matrix):
        for layer in range(self.num_layers):
            sageLayer = self.sageLayers[layer]
            features = sageLayer(features, adj_matrix)
        # print(features)
        return features


class DiffPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_of_clusters, final=False):
        super(DiffPool, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_of_clusters = num_of_clusters
        self.embedding_gnn = GraphSage(self.input_dim, self.hidden_dim, num_layer=1)
        self.assignment_gnn = GraphSage(self.input_dim, self.num_of_clusters, num_layer=1)
        self.final = final

    def forward(self, features, adj_matrix):
        z_l = self.embedding_gnn(features, adj_matrix)
        s_l = torch.softmax(self.assignment_gnn(features, adj_matrix), dim=-1)
        if self.final:
            s_l = torch.ones(s_l.shape).to(device)
        if len(features.size()) == 2:
            x_next = s_l.t().matmul(z_l)
            a_next = s_l.t().matmul(adj_matrix).matmul(s_l)
        else:
            x_next = s_l.permute(0, 2, 1).matmul(z_l)
            a_next = s_l.permute(0, 2, 1).matmul(adj_matrix).matmul(s_l)
        return x_next, a_next


class CGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_of_clusters):
        super(CGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.lr = lr
        self.gnn = GraphSage(self.input_dim, self.hidden_dim)
        self.global_graph_state = nn.ModuleList([
            GraphSage(self.input_dim, self.hidden_dim),
            DiffPool(self.hidden_dim, self.hidden_dim, num_of_clusters),
            GraphSage(self.hidden_dim, self.hidden_dim),
            DiffPool(self.hidden_dim, self.hidden_dim, 1, final=True)
        ])

        # self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, features, adj):
        for index, layer in enumerate(self.global_graph_state):
            if isinstance(layer, GraphSage):
                features = layer(features, adj)
                if index == 0:
                    node_embs = features
            elif isinstance(layer, DiffPool):
                features, adj = layer(features, adj)
        return features, node_embs


class Actor(nn.Module):
    def __init__(self, input_dim, g_hidden_dim, state_dim, hidden_dim, action_dim, max_action=1.):
        super(Actor, self).__init__()

        self.gnn_out = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)
        self.gnn_in = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)

        self.l1 = nn.Linear(state_dim * 2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, features, adj_out, adj_in):

        graph_out, node_out = self.gnn_out(features, adj_out)
        graph_in, node_in = self.gnn_in(features, adj_in)
        # print(graph_in)
        if len(features.size()) == 3:
            state_out = graph_out.matmul(node_out.permute(0, 2, 1))
            state_in = graph_in.matmul(node_in.permute(0, 2, 1))
        else:
            state_out = graph_out.matmul(node_out.t())
            state_in = graph_in.matmul(node_in.t())

        state_out = F.normalize(state_out, dim=-1)
        state_in = F.normalize(state_in, dim=-1)
        state = torch.cat([state_out, state_in], -1)

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, input_dim, g_hidden_dim, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.gnn_out_1 = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)
        self.gnn_in_1 = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)
        self.l1 = nn.Linear(state_dim * 2 + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.gnn_out_2 = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)
        self.gnn_in_2 = CGNN(input_dim, g_hidden_dim, num_of_clusters=16)
        self.l4 = nn.Linear(state_dim * 2 + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, features, adj_out, adj_in, action):
        graph_out_1, node_out_1 = self.gnn_out_1(features, adj_out)
        graph_in_1, node_in_1 = self.gnn_in_1(features, adj_in)
        if len(features.size()) == 3:
            state_out_1 = graph_out_1.matmul(node_out_1.permute(0, 2, 1))
            state_in_1 = graph_in_1.matmul(node_in_1.permute(0, 2, 1))
        else:
            state_out_1 = graph_out_1.matmul(node_out_1.t())
            state_in_1 = graph_in_1.matmul(node_in_1.t())

        state_out_1 = F.normalize(state_out_1, dim=-1)
        state_in_1 = F.normalize(state_in_1, dim=-1)
        sa_1 = torch.cat([state_out_1, state_in_1, action], -1)

        q1 = F.relu(self.l1(sa_1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        graph_out_2, node_out_2 = self.gnn_out_2(features, adj_out)
        graph_in_2, node_in_2 = self.gnn_in_2(features, adj_in)
        if len(features.size()) == 3:
            state_out_2 = graph_out_2.matmul(node_out_2.permute(0, 2, 1))
            state_in_2 = graph_in_2.matmul(node_in_2.permute(0, 2, 1))
        else:
            state_out_2 = graph_out_2.matmul(node_out_2.t())
            state_in_2 = graph_in_2.matmul(node_in_2.t())
        
        state_out_2 = F.normalize(state_out_2, dim=-1)
        state_in_2 = F.normalize(state_in_2, dim=-1)
        sa_2 = torch.cat([state_out_2, state_in_2, action], -1)

        q2 = F.relu(self.l4(sa_2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, features, adj_out, adj_in, action):
        graph_out, node_out = self.gnn_out_1(features, adj_out)
        graph_in, node_in = self.gnn_in_1(features, adj_in)
        if len(features.size()) == 3:
            state_out = graph_out.matmul(node_out.permute(0, 2, 1))
            state_in = graph_in.matmul(node_in.permute(0, 2, 1))
        else:
            state_out = graph_out.matmul(node_out.t())
            state_in = graph_in.matmul(node_in.t())

        state_out = F.normalize(state_out, dim=-1)
        state_in = F.normalize(state_in, dim=-1)
        sa = torch.cat([state_out, state_in, action], -1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class GAC(nn.Module):
    def __init__(self, state_dim, g_hidden_dim, fc_hidden_dim, action_dim, max_action, discount, tau,
                 policy_noise, noise_clip, policy_freq, actor_lr, critic_lr, num_of_users):
        super(GAC, self).__init__()
        state_dim = int(state_dim / num_of_users)
        self.actor = Actor(state_dim, g_hidden_dim, num_of_users, fc_hidden_dim, action_dim, max_action=max_action).to(
            device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, g_hidden_dim, num_of_users, fc_hidden_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.num_of_users = num_of_users

        self.adjs_out = None
        self.adjs_in = None

        self.name = "GAC"

    def set_adjs(self, adjs_out, adjs_in):
        self.adjs_out = torch.FloatTensor(adjs_out).to(device)
        self.adjs_in = torch.FloatTensor(adjs_in).to(device)

    def select_action(self, features):
        return self.actor(features.reshape(self.num_of_users, -1), self.adjs_out, self.adjs_in).flatten()


    def soft_update(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def train(self, replaybuffer, batch_size):
        self.total_it += 1
        features, actions, next_features, rewards, dones = replaybuffer.sample(batch_size)

        features = features.reshape((batch_size, self.num_of_users, -1))
        next_features = next_features.reshape((batch_size, self.num_of_users, -1))
        adjs_out = self.adjs_out.unsqueeze(0)
        adjs_in = self.adjs_in.unsqueeze(0)
        actions = actions.reshape((batch_size, 1, -1))
        rewards = rewards.reshape((batch_size, 1, 1))
        dones = dones.reshape((batch_size, 1, 1))


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_features, adjs_out, adjs_in) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_features, adjs_out, adjs_in, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = rewards + dones * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(features, adjs_out, adjs_in, actions)
        #
        # # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #
        # # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #
        # # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(features, adjs_out, adjs_in, self.actor(features, adjs_out, adjs_in)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update()
