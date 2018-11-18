
import pickle, os

import collections
ExtraData = collections.namedtuple('ExtraData', ('duplicate'))
ExtraDataSet = collections.namedtuple('ExtraDataSet', ('train', 'test'))
ExtraData2 = collections.namedtuple('ExtraData', ())

import numpy as np

class load_data(object):
  def __init__(self):
      pass
  def __call__(self, load_from_saved=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if(load_from_saved==True):
        print("Load from saved...")
        dataset2 = pickle.load(open("MAIN_nn_dataset.pickle", "rb"))

        def reshaping(i):
            i = np.expand_dims(i, axis=2)
            iss = i.shape
            print(iss)
            iss = [iss[0], np.int(iss[1]/2), 2]
            print(iss)
            i = np.reshape(i,iss)
            return i
        extra_data = ExtraData(ExtraDataSet([],[]))
        extra_data.duplicate.train.append(reshaping(dataset2["train_data"]))
        extra_data.duplicate.test.append(reshaping(dataset2["test_data"]))
        extra_data = ExtraData2()
        dataset = [reshaping(dataset2["train_data"]), reshaping(dataset2["train_labels"]), reshaping(dataset2["test_data"]), reshaping(dataset2["test_labels"])] + [extra_data]
        print("Finished load from saved...")
        return dataset
    return dataset
