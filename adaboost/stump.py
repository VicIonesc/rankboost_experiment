from adaboost.mydataset import Data


class Stump:

    def __init__(self,threshold):
        self.threshold = threshold

    def predict_one(self,x):
        if x > self.threshold:
            return 1
        return -1

    def predict_all(self,values):
        predictions = []

        for value in values:
            prediction = self.predict_one(value)
            predictions.append(prediction)

        return predictions

  #  provider = Data()

  #  x = provider.x
  #  y = provider.y

  #  prediction = 0
