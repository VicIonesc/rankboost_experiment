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

    error = 0

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

new_weights = []

for i in range (len(data.y)):
    updated_weight = data.weights[i] * math.exp(-alpha * data.y[i] * best_predictions[i])
    new_weights.append(updated_weight)

print("new weights before normalization:", new_weights)

total = sum(new_weights)

for i in range(len(new_weights)):
    new_weights[i] = new_weights[i] / total

print("new weights after normalization:", new_weights)
print("sum:", sum(new_weights))
