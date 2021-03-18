from abc import ABC, abstractmethod
import socket


class Player(ABC):
    """Part of the gym environment, need to handle low-level issues with java end interaction
    some of the member function are deprecated because of contianing high-level operations (Moved to
    microrts.algo.agents)
    """
    conn = None
    type = None
    port = None
    _client_ip = None
    id = None

    # very long memory
    brain = None

    # long memory
    _memory = None

    # short memories
    last_actions = None
    units_on_working = {}
    player_actions = None   # used to interact with env

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __init__(self, pid, client_ip, port, memory_size=10000):
        self.id = pid
        self.port = port
        self._client_ip = client_ip
    
    # def forget(self):
    #     """Forget what short memories stored, remain the long and very long 's
    #     """
    #     self.last_actions = None
    #     self.units_on_working.clear()
    #     self.player_actions = None

    def join(self):
        """
        hand shake with java end
        """
        server_socket = socket.socket()
        server_socket.bind((self._client_ip, self.port))
        server_socket.listen()
        print("Player{} Wait for Java client connection...".format(self.id))
        self.conn, address_info = server_socket.accept()
        self.greetings()

    def greetings(self):
        print("Player{}: Send welcome msg to client...".format(self.id))
        self._send_msg("Welcome msg sent!")
        print(self._recv_msg())

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, action):
        """
        Do some action according to action_sampler in the env together with other players
        """
        pass

    @abstractmethod
    def observe(self):
        """
        observe the feedback from the env
        """
        pass

    def expect(self):
        """Expecting and waiting for the msg from environment
        
        Returns:
            str -- the msg received from remote
        """
        return self._recv_msg()

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occurred: ", err)
        # return self.conn.recv(65536).decode('utf-8')

    def _recv_msg(self):
        return self.conn.recv(65536).decode('utf-8')

