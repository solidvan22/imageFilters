import time
import math

def getMillis():
    return math.trunc(time.time()*1000)

def getWithDefault(m,k,d):
    try:
        return m[k]
    except:
        return d
