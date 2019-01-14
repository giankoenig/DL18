import train_cifar
import eval_wrn40_2

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials

# define parameter ranges

def wrn40_2(args):
    wrn40_2_accuracy = eval_wrn40_2.train_wrn40_2(args)
    return wrn40_2_accuracy

trials = Trials()
sspace = [hp.uniform('x1', -100, 100), 
          hp.uniform('x2', -100, 100),
          hp.uniform('x3', -100, 100)] 

best = fmin(wrn40_2,
    space=sspace,    
    algo=tpe.suggest,
    max_evals=50,
    trials=trials)

print(best)

losses = []
vals = []
for t in trials.trials:
    losses.append(t['result']['loss'])
    vals.append(t['misc']['vals']['x1'])

plt.plot(losses)
plt.show()