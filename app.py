from flask import Flask
from test import Test

app = Flask(__name__)

@app.route("/")
def predict():
    test = Test()
    return test.predict()