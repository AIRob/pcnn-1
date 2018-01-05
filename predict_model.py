#encoding:UTF-8

import os
import numpy as np

class Predict(object):
    
    def __init__(self, model, database, setting):
        self.model = model
        self.database = database
        self.signals = []
        
    def _get_labels(self):
        raise NotImplementedError
        
    def _get_signals(self):
        # 获取数据集对应的预测labels
        raise NotImplementedError
    
    def _get_profit(self):
        raise NotImplementedError
    
    def show_profit(self):
        raise NotImplementedError
    

class PcnnPredict(Predict):
    
    def __init__(self, model, setting):
        super(PcnnPredict, self).__init__(model, setting)
        
    def _get_labels(self):
        data = self.database.get_next_batch()
        results = self.model.predict(data)
        results = results.reshape(-1).tolist()
        self.signals += results
        
    def _get_signals(self):
        pass
    
    def _get_profit(self):
        pass

if __name__ == '__main__':
    PcnnPredict(1, 2)