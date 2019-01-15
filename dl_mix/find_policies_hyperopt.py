import train_cifar

from hyperopt import hp, fmin, tpe, Trials

import numpy as np
import matplotlib.pyplot as plt
import pickle

# available policies: Section 2.2 in https://github.com/hyperopt/hyperopt/wiki/FMin
sspace1 = hp.choice('transformations', [
	{'type': 'ShearX', 'Prob': hp.uniform('Prob_ShearX', 0, 1), 'Mag': hp.uniform('Mag_ShearX', -0.3, 0.3), },
	{'type': 'TranslateX', 'Prob': hp.uniform('Prob_TranslateX', 0, 1),'Mag': hp.uniform('Mag_TranslateX', -150,150),},
	{'type': 'ShearY', 'Prob': hp.uniform('Prob_ShearY', 0, 1),'Mag': hp.uniform('Mag_ShearY', -0.3, 0.3), },
	{'type': 'TranslateY', 'Prob': hp.uniform('Prob_TranslateY', 0, 1),'Mag': hp.uniform('Mag_TranslateY', -150,150),},
	{'type': 'Rotate', 'Prob': hp.uniform('Prob_Rotate', 0, 1),'Mag': hp.uniform('Mag_Rotate', -30, 30),},
	{'type': 'AutoContrast', 'Prob': hp.uniform('Prob_AutoContrast', 0, 1),'Mag': hp.randint('Mag_AutoContrast', 1),},
	{'type': 'Invert', 'Prob': hp.uniform('Prob_Invert', 0, 1),'Mag': hp.randint('Mag_Invert', 1),},
	{'type': 'Equalize', 'Prob': hp.uniform('Prob_Equalize', 0, 1),'Mag': hp.randint('Mag_Equalize', 1),},
 	{'type': 'Solarize', 'Prob': hp.uniform('Prob_Solarize', 0, 1),'Mag': hp.uniform('Mag_Solarize', 0, 256),},
 	{'type': 'Polarize', 'Prob': hp.uniform('Prob_Polarize', 0, 1),'Mag': hp.uniform('Mag_Polarize', 4, 8),},
 	{'type': 'Constrast','Prob': hp.uniform('Prob_Constrast', 0, 1),'Mag': hp.uniform('Mag_Constrast', 0.1, 1.9),},
 	{'type': 'Color', 'Prob': hp.uniform('Prob_Color', 0, 1),'Mag': hp.uniform('Mag_Color', 0.1, 1.9),},
 	{'type': 'Brightness', 'Prob': hp.uniform('Prob_Brightness', 0, 1),'Mag': hp.uniform('Mag_Brightness', .1, 1.9),},
	{'type': 'Sharpness', 'Prob': hp.uniform('Prob_Sharpness', 0, 1),'Mag': hp.uniform('Mag_Sharpness', 0.1, 1.9),},
 	{'type': 'Cutout', 'Prob': hp.uniform('Prob_Cutout', 0, 1),'Mag': hp.uniform('Mag_Cutout', 0, 60),},
	{'type': 'SamplePairing', 'Prob': hp.uniform('Prob_SamplePairing', 0, 1),'Mag': hp.uniform('Mag_SamplePairing', 0, 0.4),},])

sspace2 = hp.choice('policies', [
	{'sub_policy_0': 'Invert_Contrast',
		'Prob1': hp.uniform('Prob_Invert', 0, 1),
		'Mag1': hp.randint('Mag_Invert', 1),
		'Prob2': hp.uniform('Prob_Constrast', 0, 1),
		'Mag2': hp.uniform('Mag_Constrast', 0.1, 1.9),},
	{'sub_policy_1': 'Rotate_TranslateX',
		'Prob1': hp.uniform('Prob_Rotate', 0, 1),
		'Mag1': hp.uniform('Mag_Rotate', -30, 30),
		'Prob2': hp.uniform('Prob_TranslateX', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateX', -150,150),},
	{'sub_policy_2': 'Sharpness_Sharpness',
		'Prob1': hp.uniform('Prob_Sharpness1', 0, 1),
		'Mag1': hp.uniform('Mag_Sharpness1', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Sharpness2', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness2', 0.1, 1.9),},
	{'sub_policy_3': 'ShearY_TranslateY',
		'Prob1': hp.uniform('Prob_ShearY', 0, 1),
		'Mag1': hp.uniform('Mag_ShearY', -0.3, 0.3),
		'Prob2': hp.uniform('Prob_TranslateY', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY', -150,150),},
	{'sub_policy_4': 'AutoContrast_Equalize',
		'Prob1': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast', 1),
		'Prob2': hp.uniform('Prob_Equalize', 0, 1),
		'Mag2': hp.randint('Mag_Equalize', 1),},
	{'sub_policy_5': 'ShearY_TranslateY',
		'Prob1': hp.uniform('Prob_ShearY', 0, 1),
		'Mag1': hp.uniform('Mag_ShearY', -0.3, 0.3),
		'Prob2': hp.uniform('Prob_TranslateY', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY', -150,150),},
	{'sub_policy_6': 'Color_Brightness',
		'Prob1': hp.uniform('Prob_Color', 0, 1),
		'Mag1': hp.uniform('Mag_Color', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Brightness', 0, 1),
		'Mag2': hp.uniform('Mag_Brightness', .1, 1.9),},
	{'sub_policy_7': 'Sharpness_Brightness',
		'Prob1': hp.uniform('Prob_Sharpness', 0, 1),
		'Mag1': hp.uniform('Mag_Sharpness', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Brightness', 0, 1),
		'Mag2': hp.uniform('Mag_Brightness', .1, 1.9),},
	{'sub_policy_8': 'Equalize_Equalize',
		'Prob1': hp.uniform('Prob_Equalize1', 0, 1),
		'Mag1': hp.randint('Mag_Equalize1', 1),
		'Prob2': hp.uniform('Prob_Equalize2', 0, 1),
		'Mag2': hp.randint('Mag_Equalize2', 1),},
	{'sub_policy_9': 'Contrast_Sharpness',
		'Prob1': hp.uniform('Prob_Constrast', 0, 1),
		'Mag1': hp.uniform('Mag_Constrast', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Sharpness', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness', 0.1, 1.9),},
	{'sub_policy_10': 'Color_TranslateX',
		'Prob1': hp.uniform('Prob_Color', 0, 1),
		'Mag1': hp.uniform('Mag_Color', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_TranslateX', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateX', -150,150),},
	{'sub_policy_11': 'Equalize_AutoContrast',
		'Prob1': hp.uniform('Prob_Equalize', 0, 1),
		'Mag1': hp.randint('Mag_Equalize', 1),
		'Prob2': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast', 1),},
	{'sub_policy_12': 'TranslateY_Sharpness',
		'Prob1': hp.uniform('Prob_TranslateY', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY', -150,150),
		'Prob2': hp.uniform('Prob_Sharpness', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness', 0.1, 1.9),},
	{'sub_policy_13': 'Brightness_Color',
		'Prob1': hp.uniform('Prob_Brightness', 0, 1),
		'Mag1': hp.uniform('Mag_Brightness', .1, 1.9),
		'Prob2': hp.uniform('Prob_Color', 0, 1),
		'Mag2': hp.uniform('Mag_Color', 0.1, 1.9),},
	{'sub_policy_14': 'Solarize_Invert',
		'Prob1': hp.uniform('Prob_Polarize', 0, 1),
		'Mag1': hp.uniform('Mag_Polarize', 4, 8),
		'Prob2': hp.uniform('Prob_Invert', 0, 1),
		'Mag2': hp.randint('Mag_Invert', 1),},
	{'sub_policy_15': 'Equalize_AutoContrast',
		'Prob1': hp.uniform('Prob_Equalize', 0, 1),
		'Mag1': hp.randint('Mag_Equalize', 1),
		'Prob2': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast', 1),},
	{'sub_policy_16': 'Equalize_Equalize',
		'Prob1': hp.uniform('Prob_Equalize1', 0, 1),
		'Mag1': hp.randint('Mag_Equalize1', 1),
		'Prob2': hp.uniform('Prob_Equalize2', 0, 1),
		'Mag2': hp.randint('Mag_Equalize2', 1),},
	{'sub_policy_17': 'Color_Equalize',
		'Prob1': hp.uniform('Prob_Color', 0, 1),
		'Mag1': hp.uniform('Mag_Color', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Equalize', 0, 1),
		'Mag2': hp.randint('Mag_Equalize', 1),},
	{'sub_policy_18': 'AutoContrast_Solarize',
		'Prob1': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast', 1),
		'Prob2': hp.uniform('Prob_Polarize', 0, 1),
		'Mag2': hp.uniform('Mag_Polarize', 4, 8),},
	{'sub_policy_19': 'Brightness_Color',
		'Prob1': hp.uniform('Prob_Brightness', 0, 1),
		'Mag1': hp.uniform('Mag_Brightness', .1, 1.9),
		'Prob2': hp.uniform('Prob_Color', 0, 1),
		'Mag2': hp.uniform('Mag_Color', 0.1, 1.9),},
	{'sub_policy_20': 'Solarize_AutoContrast',
		'Prob1': hp.uniform('Prob_Polarize', 0, 1),
		'Mag1': hp.uniform('Mag_Polarize', 4, 8),
		'Prob2': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast', 1),},
	{'sub_policy_21': 'TranslateY_TranslateY',
		'Prob1': hp.uniform('Prob_TranslateY1', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY1', -150,150),
		'Prob2': hp.uniform('Prob_TranslateY2', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY2', -150,150),},
	{'sub_policy_22': 'AutoContrast_Solarize',
		'Prob1': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast', 1),
		'Prob2': hp.uniform('Prob_Polarize', 0, 1),
		'Mag2': hp.uniform('Mag_Polarize', 4, 8),},
	{'sub_policy_23': 'Equalize_Invert',
		'Prob1': hp.uniform('Prob_Equalize', 0, 1),
		'Mag1': hp.randint('Mag_Equalize', 1),
		'Prob2': hp.uniform('Prob_Invert', 0, 1),
		'Mag2': hp.randint('Mag_Invert', 1),},
	{'sub_policy_24': 'TranslateY_AutoContrast',
		'Prob1': hp.uniform('Prob_TranslateY', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY', -150,150),
		'Prob2': hp.uniform('Prob_AutoContrast', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast', 1),},])

# create search space
# 1 policy consists of 5 sub-policies (from policies) with two hyper-parameters (probabilty + magnitude)
# 


# Test SSPEACE
def wrn40_2(args):
	x1 = args[0]
	x2 = args[1]
	x3 = args[2]
	a = 3
	b = 100
	acc = (a - x1)**2 + b*(x2 - x1**2)**2+x3**2

	print(acc)

	return acc

sspace = [hp.uniform('x1', -100, 100), 
          hp.uniform('x2', -100, 100),
          hp.uniform('x3', -100, 100)]  

x1 = np.arange(-4, 4, 0.1)
x2 = np.arange(-4, 4, 0.1)
x3 = np.arange(-4, 4, 0.1)

trials = Trials()


def run_trials():

	# model_name = 'wrn40_2_curr_epoch_{}.hyperopt'.format(curr_epoch)
	# try:  # try to load an already saved trials object, and increase the max
	# 	trials = pickle.load(open(model_name, "rb"))
	# 	print("Found saved Trials! Loading...")
	# 	max_trials = len(trials.trials) + trials_step
	# 	print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
	# except:  # create a new trials object and start searching
	# 	trials = Trials()

	best = fmin(wrn40_2,
    	space=sspace,    
    	algo=tpe.suggest,
    	max_evals=500,
    	trials=trials)

	print('Best: ', best)

	# # save the trials object
 #    with open(model_name, "wb") as f:
 #        pickle.dump(trials, f)

while True:
    run_trials()

