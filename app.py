from flask import Flask, request, jsonify, render_template
from test import Test

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict")
def predict():
    test = Test()
    predictedvalue = test.predict()
    return render_template('index.html', prediction_text='Prediction $ {}'.format(predictedvalue[0][0]))

    #return predictedvalue[0][0]

if __name__ == '__main__':
	app.run(debug=True)
