from fredapi import Fred

class FredWrapper:
    max_results_per_requeset = 1000
    def __init__(self, api_key) -> None:
        self.client = Fred(api_key = api_key)
    
    def get_series(self, series_id, **kwargs):
        return self.client.get_series(series_id)
    
    def get_series_info(self, series_id, **kwargs):
        return self.client.get_series_info(series_id)