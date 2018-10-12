from queue import Queue
from utils.generic import getWithDefault, getMillis

def buildQueues(queuesConf):
    queues = {}
    for conf in queuesConf:
        if getWithDefault(conf,"activate",1):
            q = Queue(maxsize=conf["depth"])
            queues[conf["name"]] = q

    return queues
