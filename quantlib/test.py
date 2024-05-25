from quantlib.datapoller.master import DataPoller
from pprint import pprint

def main():
    data_master = DataPoller()
    data = data_master.macro.get_series(series_id = "GDP")
    pprint(data)

main()