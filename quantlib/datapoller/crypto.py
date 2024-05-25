from quantlib.datapoller.base import BasePoller


class Crypto(BasePoller):
    
    def get_ohlcv(self, **kwargs):
        return self.pollers[kwargs['src']].get_ohlcv(**kwargs)
