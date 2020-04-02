from flask import Flask, request, jsonify, render_template
from test import Test

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    test = Test()
    predictedvalue = test.predict()
    return jsonify(predictedvalue)



if __name__ == '__main__':
	app.run(debug=True)