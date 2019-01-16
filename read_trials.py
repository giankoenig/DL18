import pickle
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from os import walk
import os

trialfiles = []
for (dirpath, dirnames, filenames) in walk('./trials/'):
  trialfiles.append(filenames)

for filename in trialfiles[0]:
  print(filename)
  path = os.path.join('./trials',filename)

  trials = pickle.load(open(path, "rb"))

  best_trials = sorted(trials.trials, key=lambda x: x['result']['loss'], reverse=False)
  print(best_trials[0]['result']['loss'])
  print(best_trials[0]['misc']['vals'])
