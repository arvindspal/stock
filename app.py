from flask import Flask
from test import Test

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    test = Test()
    return test.predict()






if __name__ == '__main__':
	app.run(debug=True)
