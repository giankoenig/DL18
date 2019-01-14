import train_cifar

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials

# define parameter ranges

def wrn40_2(args):
	x1 = args[0]
	x2 = args[1]
	x3 = args[2]
	a = 3
	b = 100
	acc = (a - x1)**2 + b*(x2 - x1**2)**2+x3**2

	test = train_cifar.main(args)
	print(acc)

	return acc

trials = Trials()
# available policies
policies = [hp.uniform('ShearX', -0.3, 0.3), 
			hp.uniform('TranslateX', -150,150),
			hp.uniform('Rotate', -30, 30),
			hp.randint('AutoContrast', 1),
			hp.randint('Invert', 1),
        	hp.randint('Equalize', 1),
         	hp.uniform('Solarize', 0, 256),
         	hp.uniform('Polarize', 4, 8),
         	hp.uniform('Constrast', 0.1, 1.9),
         	hp.uniform('Color', 0.1, 1.9),
         	hp.uniform('Brightness', 0.1, 1.9),
        	hp.uniform('Sharpness', 0.1, 1.9),
         	hp.uniform('Cutout', 0, 60),
        	hp.uniform('SamplePairing', 0, 0.4)] 
# create search space
# 1 policy consists of 5 sub-policies (from policies) with two hyper-parameters (probabilty + magnitude)
# 

sspace = [hp.uniform('x1', -100, 100), 
          hp.uniform('x2', -100, 100),
          hp.randint('x3', 10)]  

x1 = np.arange(-4, 4, 0.1)
x2 = np.arange(-4, 4, 0.1)
x3 = np.arange(-4, 4, 0.1)

best = fmin(wrn40_2,
    space=sspace,    
    algo=tpe.suggest,
    max_evals=500,
    trials=trials)

print(best)

losses = []
vals = []
for t in trials.trials:
    losses.append(t['result']['loss'])
    vals.append(t['misc']['vals']['x1'])

plt.plot(losses)
plt.show()