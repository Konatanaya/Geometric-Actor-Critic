import argparse
import time

import pyhocon
import GAC, TD3, ReplayBuffer
import torch
import numpy as np
import os
import environment
import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_parser():
    parser = argparse.ArgumentParser(description="Hyper Parameters")
    parser.add_argument('-d', '--dataset', type=str, default='twitter')
    parser.add_argument('-b', '--budget', type=int, default=20)
    parser.add_argument('-p', '--policy', type=str, default="TD3")
    parser.add_argument('-s', '--seed', type=int, default=321)
    parser.add_argument('-ep', '--episodes', type=int, default=50000)
    parser.add_argument('-ee', '--exp_episodes', type=int, default=1000)
    parser.add_argument('-ts', '--time_steps', type=int, default=20)
    parser.add_argument('-ef', '--eval_freq', type=int, default=100)
    parser.add_argument('-evat', '--evaluation_timesteps', type=int, default=150)
    parser.add_argument('-rb', '--replay_size', type=int, default=1e5)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)

    parser.add_argument('-t', '--train', action='store_true', default=False)
    parser.add_argument('-eva', '--evaluation', action='store_true', default=False)

    args = parser.parse_args()

    parser.print_help()
    return args


def eval_policy(model, args, eval_timesteps=20, filename=None):
    eval_env = environment.Simulation(args.dataset, 4, args.budget)
    eval_reward = 0.
    features, active_rate, active_num = eval_env.reset()
    active_nums = [active_num[0]]
    for time_step in range(1, eval_timesteps + 1):
        action = model.select_action(features).detach()
        features, rewards, active_rate, active_num, remaining_budget = eval_env.step(action)
        eval_reward += rewards[0]
        active_nums.append(active_num[0])

    if filename is not None:
        print(active_nums)
        np.save(f"./Results/{filename}", active_nums)

    return eval_reward, active_nums


def train_model(args, model, replay_buffer, file_name):
    env = environment.Simulation(args.dataset, 4, args.budget)
    evaluations = []
    starttime = time.time()
    for ep in range(0, args.episodes):
        episode_reward = 0.
        features, active_rate, active_num = env.reset()
        active_nums = [active_num[0]]
        for time_step in range(1, args.time_steps + 1):
            if ep < args.exp_episodes:
                action = env.sample_action()
            else:
                action = torch.clip(model.select_action(features) + torch.normal(-active_rate[0], 1, size=(1, env.num_of_user), device=device), -1, 1).flatten().detach()

            features_, rewards, active_rate, active_num, remaining_budget = env.step(action)

            done = 0 if time_step == args.time_steps else model.discount
            replay_buffer.add(features.cpu().data.numpy().flatten(),
                              action.cpu().data.numpy().flatten(),
                              rewards[0],
                              done)
            features = features_

            episode_reward += rewards[0]
            active_nums.append(active_num[0])
            if ep >= args.exp_episodes:
                model.train(replay_buffer, batch_size)
        if ep % 50 == 0 and ep < args.exp_episodes:
            print(f"Episode: {ep} | Reward: {episode_reward:.3f} | Final Rates: {active_nums[-1] / env.num_of_user} | Last 5 Rates: {active_nums[-5:]} | Time: {(time.time() - starttime)}s")
            starttime = time.time()
        if ep % args.eval_freq == 0 and ep >= args.exp_episodes:
            eval_reward, eval_active_num_ = eval_policy(model, args, eval_timesteps=args.time_steps)
            print(f"Eval {model.name} | Episode {ep} | Total Reward: {eval_reward:.3f} | Active num: {eval_active_num_[-5:]} | Time: {(time.time() - starttime)}s")
            evaluations.append([ep, eval_reward])
            np.save(f"./Results/{args.dataset}/eval/{file_name}", evaluations)
            if args.save_model:
                model.save(f"./Models/{args.dataset}/{file_name}")
            starttime = time.time()


if __name__ == '__main__':
    args = init_parser()
    print("---------------------------------------")
    print(f"Dataset: {args.dataset} | Budget: {args.budget} | Policy: {args.policy} ")
    print("---------------------------------------")
    file_name = f"{args.policy}_{args.budget}_{args.seed}"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dgl.seed(args.seed)

    if not os.path.exists("./Results/" + args.dataset):
        os.makedirs("./Results/" + args.dataset)

    if not os.path.exists("./Results/" + args.dataset + "/" + "eval"):
        os.makedirs("./Results/" + args.dataset + "/" + "eval")

    if not os.path.exists("./Models/" + args.dataset):
        os.makedirs("./Models/" + args.dataset)

    config = pyhocon.ConfigFactory.parse_file("hyperparameters.conf")

    env = environment.Simulation(args.dataset, 4, args.budget)

    # State
    state_dim = 5

    replay_size = int(args.replay_size)
    batch_size = args.batch_size
    num_of_users = env.num_of_user

    # Init models
    if args.policy == "TD3":
        hyperparameters = config.TD3
        hyperparameters["state_dim"] = state_dim * num_of_users
        hyperparameters["action_dim"] = num_of_users
        model = TD3.TD3(**hyperparameters)
    elif args.policy == "GAC":
        hyperparameters = config.GAC
        hyperparameters["state_dim"] = state_dim * num_of_users
        hyperparameters["action_dim"] = num_of_users
        hyperparameters["num_of_users"] = num_of_users
        model = GAC.GAC(**hyperparameters)
        model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)

    if args.policy in ["TD3",  "GAC"]:
        replay_buffer = ReplayBuffer.ReplayBuffer(hyperparameters["state_dim"], hyperparameters["action_dim"], max_size=replay_size)
    if args.train:
        train_model(args, model, replay_buffer, file_name)
    if args.evaluation:
        model.load(f"./Models/{args.dataset}/{file_name}")
        path = f"{args.dataset}/performance/{file_name}"
        eval_policy(model, args, eval_timesteps=args.evaluation_timesteps, filename=path)
