# from rts_wrapper.datatypes import config
# from rts_wrapper.envs import MicroRts
import gym
from multiprocessing import Process, Pool
from threading import Thread
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .rts_base_env import BaseEnv
from .player import Player
from .player_zt import PlayerZT
from .player_jin import PlayerJin
from .utils import get_available_port, network_simulator
from .datatypes import Observations
from dacite import from_dict


class BattleEnv(BaseEnv):
    player1 = None
    player2 = None
    players = []

    @property
    def players_num(self):
        return len(self.players)
    
    def get_players(self):
        return self.players

    def __init__(self, config):
        super(BattleEnv, self).__init__(config)
        self.DEBUG = False
        self._counting_players()

        if not self.DEBUG:
            Thread(target=self.start_client).start()
        self._players_join()

    def _counting_players(self):
        player = self._produce_player(0, self.ai1_type, self.socket_ai1_type)
        if player is not None:
            self.player1 = player
            self.players.append(player)

        player = self._produce_player(1, self.ai2_type, self.socket_ai2_type)
        if player is not None:
            self.player2 = player
            self.players.append(player)

    def _produce_player(self, player_id, ai_type, socket_ai_type):
        player = None
        if ai_type.startswith("socket") and socket_ai_type == "ZT":
            player = PlayerZT(player_id, self.config.client_ip, 9898 if self.DEBUG else get_available_port())
            self._add_commands("--port" + str(1 + player.id), str(player.port))
        elif ai_type.startswith("socket") and socket_ai_type == "Jin":
            player = PlayerJin(player_id, self.config.client_ip, 9898 if self.DEBUG else get_available_port())
            self._add_commands("--port" + str(1 + player.id), str(player.port))
        return player

    def _players_join(self):
        for player in self.players:
            player.join()

    @staticmethod
    def _obs2dataclass(obs):
        return Observations(
                    observation=obs[0],
                    reward=obs[1],
                    done=obs[2],
                    info=obs[3]
            )

    # TODO: get
    def step(self, actions):
        """
        local env should using Threads to avoid dead lock, while remote needn't
        """
        import time
        for i in range(self.players_num):
            self.players[i].act(actions[i])     #send actions to java end
            # time.sleep(.1)  
                                                #receive obs from java end
        signals_res = [self._obs2dataclass(player.observe()) for player in self.players]
        return signals_res

    def reset(self, **kwargs):
        """Connect to the environment by handshakes, and refresh the players' short memories
        
        Returns:
            [type] -- [description]
        """
        # print("env reset")
        signals = [self._obs2dataclass(player.reset()) for player in self.players]
        
        # for p in self.players:
        #     p.forget()

        return signals

    def get_winner(self):
        """
        this function must come after the done is true (i.e. game is over)
        -1: tie
        0: player 0
        1: player 1
        :return:
        """
        for p in self.players:
            res = p.expect()
        # print("winner is :", res)
        return int(res)


def main():
    env = gym.make("LargeMapTest-v0")
    players = env.players

    for _ in range(env.max_episodes):
        obses = env.reset()  # p1 and p2 reset
        while not obses[0].done:
            actions = []
            for i in range(len(players)):
                # players[i].think(obses[i])
                actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses = env.step(actions)
            # print(obses)
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)

# res = []
# def callback(result):
#     res.append(result)


if __name__ == '__main__':
    # p = ThreadPool(3)
    # print(p.apply_async(func=sum, args=([12,3],)).get())
    main()   
