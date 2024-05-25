from quantlib.datapoller.base import BasePoller
from quantlib.datapoller.utils import poller, ts_poller

class Macro(BasePoller):
    @poller
    def get_series(self,**kwargs):
        return self.pollers[kwargs['src']].get_series(**kwargs)
    
    @poller
    def get_series_info(self, **kwargs):
        return self.pollers[kwargs['src']].get_series_info(**kwargs)
    
