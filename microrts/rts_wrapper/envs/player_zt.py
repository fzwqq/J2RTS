from .utils import signal_wrapper, network_action_translator, pa_to_jsonable, action_sampler_v1, get_action_index

from .player import Player


class PlayerZT(Player):
    """Part of the gym environment, need to handle low-level issues with java end interaction
    some of the member function are deprecated because of contianing high-level operations (Moved to
    microrts.algo.agents)
    """

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __init__(self, pid, client_ip, port, memory_size=10000):
        super(PlayerZT, self).__init__(pid, client_ip, port, memory_size)

    # def forget(self):
    #     """Forget what short memories stored, remain the long and very long 's
    #     """
    #     self.last_actions = None
    #     self.units_on_working.clear()
    #     self.player_actions = None

    def reset(self):
        print("Server: Send reset command...")
        self._send_msg('reset')
        # print("waiting")
        raw = self._recv_msg()
        # print("received")
        # print(raw)
        return signal_wrapper(raw)

    def act(self, action):
        """
        Do some action according to action_sampler in the env together with other players
        """
        # assert self.player_actions is not None
        assert action is not None
        action = network_action_translator(action)
        pa = pa_to_jsonable(action)
        self._send_msg(pa)

    def observe(self):
        """
        observe the feedback from the env
        """
        raw = self._recv_msg()
        # print(raw)
        return signal_wrapper(raw)


if __name__ == "__main__":
    print("OK")