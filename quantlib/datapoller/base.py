from collections import deque, defaultdict


class BasePoller:
    def __init__(self, pollers, buffer_len=1000000000, default_src="") -> None:
        self.pollers = pollers
        self.default_src = default_src
        self.stream_buffer = defaultdict(lambda: deque(maxlen=buffer_len))
