import torch
from microrts.algo.model_jin import ActorCritic2
from microrts.algo.utils import load_model2

from microrts.rts_wrapper.envs.multi_envs_jin import make_vec_envs

from microrts.algo.agents_jin import Agent2
from torch.utils.tensorboard import SummaryWriter
from microrts.algo.replay_buffer2 import ReplayBuffer2

from microrts.rts_wrapper.envs.utils import action_sampler_v2, network_simulator
from microrts.algo.ppo2 import PPO2
from microrts.algo.a2c import A2C
from microrts.rts_wrapper.envs.datatypes import Config
import microrts.settings as settings
import os
import argparse
from functools import partial

# AI added in league combat
# private static AI getAIType(String type, UnitTypeTable utt){
#     switch (type){
#         case "WorkerRush"   : return new WorkerRush(utt);
#         case "Random"       : return new RandomAI();
#         case "RandomBiased" : return new RandomBiasedAI(utt);
#         case "NaiveMCTS"    : return new NaiveMCTS(utt);
#         case "MC"           : return new MonteCarlo(utt);
#         case "UCT"          : return new UCT(utt);
#         case "ABCD"         : return new ABCD(utt);
#         case "MCTS"         : return new MLPSMCTS(utt);
#         case "WorkerDefense": return new WorkerDefense(utt);
#         case "LightRush"    : return new LightRush(utt);
#         case "WorkerRushPP" : return new WorkerRushPlusPlus(utt);
#         case "PortfolioAI"  : return new PortfolioAI(utt);
#         case "SCV"          : return new SCV(utt);
#         default             : return new PassiveAI();
#     }
# }

"""
loop:
    num_prosses_pair_actions
"""


def get_config(env_id) -> Config:
    from microrts.rts_wrapper import environments
    for registered in environments:
        if registered["id"] == env_id:
            return registered['kwargs']['config']
            # return registered['kwargs']['config'].height, registered['kwargs']['config'].width


def play(args):
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    # def memo_inserter(transitions):
    #     nonlocal T
    #     T += 1
    #     # if transitions['reward'] < 0:
    #     # print(transitions['reward'])
    #     memory.push(**transitions)

    def memo_inserter2(idx, transitions):
        # nonlocal T
        # T += 1
        # if transitions['reward'] < 0:
        # print(transitions['reward'])
        buffers[idx].push(**transitions)

    num_process = args.num_process
    envs_id = args.envs_id.split(',')
    multi_envs = [args.opponent for _ in range(num_process)]
    maps_size = []
    if num_process < len(envs_id):
        print('The envs input is larger than the number of process, error')
        return None
    else:
        print('The multi-maps training starting')
        for i, x in enumerate(envs_id):
            print(x)
            if x != "None":
                multi_envs[i] = x
                config = get_config(x)
                config.render = args.render
                config.ai2_type = args.opponent
                config.max_episodes = int(args.episodes)
                config.socket_ai1_type = args.ai1_socket
                config.socket_ai2_type = args.ai2_socket
                map_size = config.height, config.width
                maps_size.append(map_size)

    nn_path = args.model_path
    start_from_scratch = nn_path is None
    Agent2.gamma = args.gamma
    # memory = ReplayBuffer2(10000)

    if start_from_scratch:
        nn = ActorCritic2(maps_size[0])  # for any map_size, no different
    else:
        nn = load_model2(os.path.join(settings.models_dir, nn_path), map_size, args.recurrent)

    # nn.share_memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    # input()
    nn.to(device)

    league = [args.opponent for _ in range(num_process)]
    cmd_league = args.league.split(',')
    if num_process < len(cmd_league):
        print('The league input is larger than the number of process, will not use league learning')
    else:
        print("league learning staring")
        for i, x in enumerate(cmd_league):
            print(x)
            if x != "None":
                league[i] = x
    print('All leagues participated are', league)
    # input()

    envs, agents = make_vec_envs(envs_id, num_process, "spawn", nn, league=league, maps_size=maps_size)

    buffers = [ReplayBuffer2(config.max_cycles + 100) for _ in range(len(agents))]
    import time
    frames = 0
    st = time.time()
    obses_n = envs.reset()
    update_steps = 32
    # T = 1
    if args.algo == "a2c":
        algo = A2C(
            ac_model=nn,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            weight_decay=3e-6,
            log_interval=args.log_interval,
            gamma=args.gamma,
            debug=args.debug,
        )
    elif args.algo == "ppo":
        algo = PPO2(
            ac_model=nn,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            weight_decay=3e-6,
            log_interval=args.log_interval,
            gamma=args.gamma,
            debug=args.debug,
        )
    writer = SummaryWriter()
    Ts = [0 for _ in range(num_process)]
    iter_idx = 0
    epi_idx = 0
    while 1:
        time_stamp = []
        actions_n = []
        for i in range(num_process):
            action_i = []
            Ts[i] += 1
            for j in range(len(obses_n[i])):
                if sum(Ts) % 1024 == 0:
                    for k in range(num_process):
                        Ts[k] = 0
                        algo.update(buffers[k], iter_idx, callback=logger, device=device)
                    iter_idx += 1
                if not obses_n[i][j].done:
                    if args.algo == 'ppo':
                        action = agents[i][j].think(sp_ac=algo.target_net, callback=memo_inserter2, debug=args.debug,
                                                    obses=obses_n[i][j], accelerator=device, mode="train", idx=i)
                    elif args.algo == 'a2c':
                        action = agents[i][j].think(callback=memo_inserter2, debug=args.debug, obses=obses_n[i][j],
                                                    accelerator=device, mode="train", idx=i)
                else:
                    action = []  # reset
                    epi_idx += .5
                    time_stamp.append(obses_n[i][j].info["time_stamp"])
                    writer.add_scalar("rewards_per_step", agents[i][j].rewards / (obses_n[i][j].info["time_stamp"]),
                                      epi_idx)
                    writer.add_scalar("rewards", agents[i][j].rewards, epi_idx)
                    if args.algo == 'ppo':
                        agents[i][j].sum_up(sp_ac=algo.target_net, callback=memo_inserter2, debug=args.debug,
                                            obses=obses_n[i][j], accelerator=device, mode="train", idx=i)
                    elif args.algo == 'a2c':
                        agents[i][j].sum_up(callback=memo_inserter2, debug=args.debug, obses=obses_n[i][j],
                                            accelerator=device, mode="train", idx=i)
                    # buffers[i]
                    agents[i][j].forget()

                action_i.append(action)

                if (epi_idx + 1) > 10000 and (epi_idx + 1) % 1000 == 0:
                    folder = settings.models_dir
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    torch.save(nn.state_dict(),
                               os.path.join(settings.models_dir, args.saving_prefix + str(int(epi_idx)) + ".pth"))

            # if obses_n[i][0].done:
            #     print(len(buffers[i]))
            #     algo.update(buffers[i], iter_idx, callback=logger, device=device)
                # if T % (update_steps * num_process) == 0:
                #     T = 1
                #     # print('Update...')
                #     # input()
                #     algo.update(memory, iter_idx, callback=logger, device=device)
                #     iter_idx += 1

            actions_n.append(action_i)

        if time_stamp:
            writer.add_scalar("TimeStamp", sum(time_stamp) / (len(time_stamp)), epi_idx)
        obses_n = envs.step(actions_n)
        frames += 1

        if frames >= 1000:
            print("fps", frames * num_process / (time.time() - st))
            frames = 0
            st = time.time()
            # torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs-id",
        default="attackHome-v0"
    )
    parser.add_argument(
        '--model-path', help='path of the model to be loaded',
        default=None
    )
    parser.add_argument(
        '--num-process',
        type=int,
        default=4
    )
    parser.add_argument(
        '--smooth-ratio',
        type=float,
        default=.0
    )
    parser.add_argument(
        '--episodes',
        # default=10e6,
        type=int,
        default=10e4,
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=int(100),
    )
    parser.add_argument(
        '--recurrent',
        action="store_true",
        # type=bool,
        default=False,
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--value_loss_coef',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99
    )
    parser.add_argument(
        '--render',
        type=int,
        default=0
    )
    parser.add_argument(
        '--opponent',
        default="socketAI"
    )
    parser.add_argument(
        "--saving-prefix",
        default='rl',
    )
    parser.add_argument(
        "--algo",
        default='ppo',
    )
    parser.add_argument(
        "--league",
        default='None',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ai1_socket",
        default="ZT",
    )
    parser.add_argument(
        "--ai2_socket",
        default="ZT",
    )
    args = parser.parse_args()
    print(args.league)
    # input()
    torch.manual_seed(0)
    play(args)  # , nn_path=os.path.join(settings.models_dir,"rl39699.pth"))

