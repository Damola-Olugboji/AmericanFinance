from quantlib.datapoller.master import DataMaster

def main():
    data_master = DataMaster()
    start = None
    end = None
    granularity = "1d"
    data_master.crypto.get_ohlcv(ticker = "BTCUSDT", start = start, end = end, granularity = granularity, src = "binance")