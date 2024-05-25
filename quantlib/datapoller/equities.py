from quantlib.datapoller.base import BasePoller
from quantlib.datapoller.utils import poller, ts_poller

class Equities(BasePoller):
    @ts_poller
    def get_ohlcv(self, **kwargs):
        return self.pollers[kwargs['src']].get_ohlcv(**kwargs)
    
