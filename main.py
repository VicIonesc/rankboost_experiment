from adaboost.mydataset import Data
from adaboost.stump import Stump

data = Data()

stump = Stump(threshold=2)

predictions = stump.predict_all(data.x)

print("x:", data.x)
print("y:", data.y)
print("predictions:", predictions)
