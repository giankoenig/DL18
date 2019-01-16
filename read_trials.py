import pickle
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from os import walk
import os
import re

trialfiles = []
for (dirpath, dirnames, filenames) in walk('./trials/'):
  trialfiles.append(filenames)
  

  
trialfiles[0].sort()

good_policies = []
keys = [['Invert','Prob_Invert1', 'Mag_Invert1', 'Contrast', 'Prob_Constrast2', 'Mag_Constrast2'],
       ['Rotate', 'Prob_Rotate3', 'Mag_Rotate3', 'TranslateX', 'Prob_TranslateX4', 'Mag_TranslateX44'],
       ['Sharpness', 'Prob_Sharpness5', 'Mag_Sharpness5', 'Sharpness', 'Prob_Sharpness6', 'Mag_Sharpness6'],
       ['ShearY', 'Prob_Shear7', 'Mag_ShearY7', 'TranslateY', 'Prob_Translate8', 'Mag_TranslateY8'],
       ['AutoContras', 'Prob_AutoContrast9', 'Mag_AutoContrast9', 'Equalize', 'Prob_Equalize10', 'Mag_Equalize10'],
       ['ShearY', 'Prob_ShearY11', 'Mag_ShearY11', 'TranslateY', 'Prob_TranslateY12', 'Mag_TranslateY12'],
       ['Color', 'Prob_Color13', 'Mag_Color13', 'Brightness', 'Prob_Brightness14', 'Mag_Brightness14'],
       ['Sharpness', 'Prob_Sharpness15', 'Mag_Sharpness15', 'Brightness', 'Prob_Brightness16', 'Mag_Brightness16'],
       ['Equalize', 'Prob_Equalize17', 'Mag_Equalize17', 'Equalize', 'Prob_Equalize18', 'Mag_Equalize18'],
       ['Contras', 'Prob_Constrast19', 'Mag_Constrast19', 'Sharpness', 'Prob_Sharpness20', 'Mag_Sharpness20'],
       ['Color', 'Prob_Color21', 'Mag_Color21', 'TranslateX', 'Prob_TranslateX22', 'Mag_TranslateX22'],
       ['Equalize', 'Prob_Equalize23', 'Mag_Equalize23', 'AutoContrast', 'Prob_AutoContrast24', 'Mag_AutoContrast24'],
       ['TranslateY', 'Prob_TranslateY25', 'Mag_TranslateY25', 'Sharpness', 'Prob_Sharpness26', 'Mag_Sharpness26'],
       ['Brightness', 'Prob_Brightness27', 'Mag_Brightness27', 'Color', 'Prob_Color28', 'Mag_Color28'],
       ['Solarize', 'Prob_Polarize29', 'Mag_Polarize29', 'Invert', 'Prob_Invert30', 'Mag_Invert30'],
       ['Equalize', 'Prob_Equalize31', 'Mag_Equalize31', 'AutoContrast', 'Prob_AutoContrast32', 'Mag_AutoContrast32'],
       ['Equalize', 'Prob_Equalize33', 'Mag_Equalize33', 'Equalize', 'Prob_Equalize34', 'Mag_Equalize34'],
       ['Color', 'Prob_Color35', 'Mag_Color35', 'Equalize', 'Prob_Equalize36', 'Mag_Equalize36'],
       ['AutoContrast', 'Prob_AutoContrast37', 'Mag_AutoContrast37', 'Solarize', 'Prob_Polarize38', 'Mag_Polarize38'],
       ['Brightness', 'Prob_Brightness39', 'Mag_Brightness39', 'Color', 'Prob_Color40', 'Mag_Color40'],
       ['Solarize', 'Prob_Polarize41', 'Mag_Polarize41', 'AutoContrast', 'Prob_AutoContrast42', 'Mag_AutoContrast42'],
       ['TranslateY', 'Prob_TranslateY43', 'Mag_TranslateY143', 'TranslateY', 'Prob_TranslateY44', 'Mag_TranslateY44'],
       ['AutoContrast', 'Prob_AutoContrast45', 'Mag_AutoContrast45', 'Solarize', 'Prob_Polarize46', 'Mag_Polarize46'],
       ['Equalize', 'Prob_Equalize47', 'Mag_Equalize47', 'Invert', 'Prob_Invert48', 'Mag_Invert48'],
       ['TranslateY', 'Prob_TranslateY49', 'Mag_TranslateY49', 'AutoContrast', 'Prob_AutoContrast50', 'Mag_AutoContrast50'],]

i = 0
for filename in trialfiles[0]:
  print(filename)
  path = os.path.join('./trials',filename)

  trials = pickle.load(open(path, "rb"))

  best_trials = sorted(trials.trials, key=lambda x: x['result']['loss'], reverse=False)

  print('loss: {}'.format(best_trials[0]['result']['loss']))
  #print(best_trials[0]['misc']['vals'])
  sub_policy = [(keys[i][0], best_trials[0]['misc']['vals'][keys[i][1]], best_trials[0]['misc']['vals'][keys[i][2]]),
                (keys[i][3], best_trials[0]['misc']['vals'][keys[i][4]], best_trials[0]['misc']['vals'][keys[i][5]])]
  good_policies.append(sub_policy)
  print(sub_policy)
  i += 1

  
with open('optimal_policies_0.pol', "wb") as f:
		pickle.dump(good_policies, f)