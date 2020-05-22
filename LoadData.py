import tensorflow as tf
import numpy as np
import random 
class DataReader():
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size= batch_size
        with open(data_path) as f:
            d_lines= f.read().splitlines()
        
        self._data=[]
        self._labels=[]
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for i in range(vocab_size)]
            feature = line.split('<fff>')
            label, doc_id = int(feature[0]), int(feature[1])
            tokens = feature[2].split() 
            for token in tokens:
                index, value= int(token.split(":")[0]), float(token.split(":")[1])
                vector[index]= value
        
            self._data.append(vector)
            self._labels.append(label)

        self._data= np.array(self._data)
        self._labels= np.array(self._labels)

        self._num_epoch=0
        self._batch_id=0

    def next_batch(self):
        # Load data from batch and suffle them.
        start = self._batch_id * self._batch_size
        end   = start + self._batch_size
        self._batch_id +=1

        if end + self._batch_size > len(self._data):
            # khi đến batch cuối, một epoch có kích thước bằng data, nên 
            # khi chọn data set cho từng patch cần xáo chộn ngẫu nhiên rồi chọn cho đủ patch
            # dù cho trong 1 epoch có thể không chứa toàn bộ dữ liệu nhưng model có thể học dần 
            # dần thông qua các epoch
            end = len(self._data)
            self._num_epoch += 1 
            self._batch_id = 0

            # suffle data
            indices= list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels= self._data[indices], self._labels[indices]

        return self._data[start:end], self._labels[start:end]