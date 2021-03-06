from train_cifar import CifarModelTrainer
import tensorflow as tf
import shutil

tf.flags.DEFINE_string('model_name', 'wrn',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '../training', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '../data',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')

FLAGS = tf.flags.FLAGS



def main(_):
  print('starting training')
  hparams = tf.contrib.training.HParams(
      train_size=4000,
      validation_size=500,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=256,
      gradient_clipping_by_global_norm=5.0)

  hparams.add_hparam('model_name', 'wrn')
  hparams.add_hparam('num_epochs', 5)
  hparams.add_hparam('wrn_size', 32)
  hparams.add_hparam('lr', 0.1)
  hparams.add_hparam('weight_decay_rate', 5e-4)
  
  results = []
  for i in range(2):
    cifar_trainer = CifarModelTrainer(hparams)
    result = cifar_trainer.run_model()
    results.append(result)
    shutil.rmtree(FLAGS.checkpoint_dir)

  print(results)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
