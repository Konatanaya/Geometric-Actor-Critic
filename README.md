# GAC: A Deep Reinforcement Learning Model Toward User Incentivization in Unknown Social Networks

This is a PyTorch implementation of Geometric Actor-Critic (GAC). Paper link: <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705122011534">Journal</a> |
<a href="https://arxiv.org/abs/2203.09578">arXiv</a>

If you make use of this code or the GAC algorithm in your work, please cite the following paper:

    @article{WU2022GAC,
        title = {GAC: A deep reinforcement learning model toward user incentivization in unknown social networks},
        journal = {Knowledge-Based Systems},
        pages = {110060},
        year = {2022},
        issn = {0950-7051},
        doi = {https://doi.org/10.1016/j.knosys.2022.110060},
        author = {Wu, Shiqing and Li, Weihua and Bai, Quan},
    }

    @inproceedings{wu2021learning,
      title={Learning Policies for Effective Incentive Allocation in Unknown Social Networks},
      author={Wu, Shiqing and Bai, Quan and Li, Weihua},
      booktitle={Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems},
      pages={1701--1703},
      year={2021}
    }

If you make use of Agent-based Decision-Making (ADM) Model in your work, please cite the following paper:

    @inproceedings{wu2019adaptive,
      title={Adaptive incentive allocation for influence-aware proactive recommendation},
      author={Wu, Shiqing and Bai, Quan and Kang, Byeong Ho},
      booktitle={Pacific Rim International Conference on Artificial Intelligence},
      pages={649--661},
      year={2019},
      organization={Springer}
    }

## Usage
You can install all the required packages using the following command:

    $ pip install -r requirements.txt

To train GAC, you can use the command below directly. But it is recommended to specify the dataset and the budget using `-d` and
`-b`, respectively. In the sample command, `-t` means training GAC. Otherwise, you may want to use `-eva` to
evaluate GAC.

    $ python main.py -p GAC -t
    $ python main.py -p GAC -t -d twitter -b 20
    $ python main.py -p GAC -eva -d twitter -b 20

To run compared algorithms (which are not RL-based methods), use the command below:
    
    $ python main_nm.py -d twitter -b 20

To plot the result, use the command:

    $ python ploy.py -d twitter