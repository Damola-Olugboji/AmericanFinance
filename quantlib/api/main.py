from flask import Flask, jsonify, request
from quantlib.data_service.data_master import DataMaster

app = Flask(__name__)

data_master = DataMaster()
crypto_service = data_master.get_crypto_service()

@app.route("/")
def index():
    return "Welcome to the American Finance API"

@app.route("/securities/crypto/get_symbol_info/<symbol>", methods = ["GET"])
def get_crypto_symbol_info(symbol):

    data = {"avg_price": crypto_service.get_symbol_info(symbol)}
    return jsonify(data)

@app.route("/securities/crypto/get_avg_price/<symbol>", methods = ["GET"])
def get_crypto_avg_price(symbol):

    # data = {"avg_price": crypto_service.get_avg_price("binance", symbol)}
    return jsonify(crypto_service.get_avg_price("binance", symbol))

if __name__ == "__main__":
    app.run(debug = True)