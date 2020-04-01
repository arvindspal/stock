from makeprediction import makePrediction

class Test():
    def __init__(self):
        self.name = 'test'
        
    def predict(self):
        classObj = makePrediction()
        df = classObj.load_data()
        df2 = classObj.scale_data(df)
        classObj.get_test_data(df2)
        classObj.load_model()
        pred = classObj.make_prediction()
        return pred