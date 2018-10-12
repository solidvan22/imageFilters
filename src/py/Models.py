from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras.models import load_model

from utils.generic import getWithDefault, getMillis

class SSD7Model:
    def __init__(self,name,weightsPath):
        self.name = name
        self.ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.weightsPath = weightsPath
        self.model = load_model(
            self.weightsPath,
            custom_objects={
                'AnchorBoxes': AnchorBoxes,
                'compute_loss': self.ssd_loss.compute_loss
            }
        )
        #warmup
        print("WARMING UP .." + self.name)
        self.model._make_predict_function()

    def getName(self):
        return self.name

    def getModel(self):
        return self.model

def buildModels(modelsConf):
    models = {}
    for conf in modelsConf:
        if getWithDefault(conf,"activate",1):
            model = SSD7Model(conf["name"],conf["weightsPath"])
            models[model.getName()] = model

    return models
