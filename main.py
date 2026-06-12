from adaboost.mydataset import Data
from adaboost.stump import Stump
import math

data = Data()

stump = Stump(threshold=2)

predictions = stump.predict_all(data.x)

print("x:", data.x)
print("y:", data.y)
print("predictions:", predictions)

thresholds = [1,2,3,4]


best_stump = None
best_error = float("inf")
best_predictions = None

for threshold in thresholds:
    stump = Stump(threshold = threshold)
    predictions = stump.predict_all(data.x)

    error = 0.2

    for i in range(len(data.y)):
        if predictions[i] != data.y[i]:
            error += data.weights[i]

    if error < best_error:
        best_error = error
        best_stump = stump
        best_predictions = predictions

    print ("threshold:", threshold)
    print ("predictions", predictions)
    print ("error:", error)
    print ()

print ("BEST STUMP:")
print ("threshold:", best_stump.threshold)
print ("error:", best_error)
print ("predictions:", best_predictions)


alpha = 0.5 * math.log((1 - best_error) / best_error)

print ("alpha:", alpha)