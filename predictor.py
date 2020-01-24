import os
import pickle

import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

class MyPredictor(object):
  def __init__(self, model, preprocessor):
    self._model = model
    self._preprocessor = preprocessor
    self._class_names = load_iris().target_names

  def predict(self, instances, **kwargs):
    inputs = np.asarray(instances)
    preprocessed_inputs = self._preprocessor.preprocess(inputs)
    outputs = self._model.predict(preprocessed_inputs)
    if kwargs.get('probabilities'):
      return outputs.tolist()
    else:
      return [self._class_names[index] for index in np.argmax(outputs, axis=1)]

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'model.h5')
    model = tf.keras.models.load_model(model_path)

    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
      preprocessor = pickle.load(f)

    return cls(model, preprocessor)