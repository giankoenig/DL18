from train_cifar import CifarModelTrainer
import tensorflow as tf
import shutil
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from search_space import SearchSpace
import pickle
import os, errno

tf.flags.DEFINE_string('model_name', 'wrn',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '../../training', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '../../data',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')

FLAGS = tf.flags.FLAGS


def eval_wrn_40_2(args):
  hparams = tf.contrib.training.HParams(
      train_size=4000,
      validation_size=500,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=256,
      gradient_clipping_by_global_norm=5.0)

  hparams.add_hparam('model_name', 'wrn')
  hparams.add_hparam('num_epochs', 10)
  hparams.add_hparam('wrn_size', 32)
  hparams.add_hparam('lr', 0.1)
  hparams.add_hparam('weight_decay_rate', 5e-4)
    
  cifar_trainer = CifarModelTrainer(hparams)

  ops = args['sub_policy'].split('_')
  policy = [(ops[0], args['Prob2'], args['Mag1']),(ops[1], args['Prob2'], args['Mag2'])]
  result = cifar_trainer.run_model(policy)
  shutil.rmtree(FLAGS.checkpoint_dir)
  return 1-result[2]

def eval_fake(args):
  print(args)
  ops = args['sub_policy'].split('_')
  policy = [(ops[0], args['Prob2'], args['Mag1']),(ops[1], args['Prob2'], args['Mag2'])]
  print(policy)
  return 0

def run_trial(policy_nr):
  
  trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
  max_trials = 1  # initial max_trials. put something small to not have to wait
<<<<<<< HEAD
  model_name = 'eval_trials_wrn_40_2_subpolicy{:02d}.hyperopt'.format(policy_nr)
  filename = os.path.join('../../trials',model_name)
  
  if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
=======
  model_name = 'eval_trials_wrn_40_2.hyperopt'
>>>>>>> cc4e627413686c962dd969109ce839100ab38a79
  
  try:  # try to load an already saved trials object, and increase the max
	trials = pickle.load(open(filename, "rb"))
	print("Found saved Trials! Loading...")
	max_trials = len(trials.trials) + trials_step
	print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
  except:  # create a new trials object and start searching
	trials = Trials()
  best = fmin(eval_wrn_40_2,
    	space=SearchSpace()[policy_nr],    
    	algo=tpe.suggest,
    	max_evals=max_trials,
    	trials=trials)

  print('Best: ', best)

  # save the trials object
  with open(filename, "wb") as f:
    pickle.dump(trials, f)

def main(_):
  print('starting training')
  
  for sub_policy in range(25):
    for runs in range(10):
      run_trial(sub_policy)
  

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
