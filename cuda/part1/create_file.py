import math
from random import random

file_name = 'test.txt'
numElements = 16384

with open(file_name, 'w') as f:
    for _ in range(numElements):
        f.write("{0:0.2f} ".format(random() * math.pi))
    f.close()

