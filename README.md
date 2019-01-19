# Deep Learning Project 2018

Instructions on how to run the code for the project 'Using Bayesian Optimization to Find Good Augmentation Policies from Data'

<b>Get the data</b>
```shell
curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
Copy the data to `./data`

<b>Optimize policies using Bayesian Optimization on WRN-40-2 on reduced CIFAR-10 data</b>

```shell
nohup python DL18/dl_mix/eval_policy.py > hyperopt_output.txt &
```

<b>Convert optimal solution found in step 2</b>

Set name on line 61 in `./DL18/read_trials.py`

```shell
python DL18/read_trials.py
```

<b>Read in optimal solution in main program</b>

- Set same name as chosen in step 3 on line 4 in `./DL18/autoaugment/hyperopt_policies.py`
- Ensure line 51 is active in `./DL18/autoaugment/data_utils.py`


<b>Train WRN-28-10 on full CIFAR-10 data with optimized augmentation policies</b>

```shell
nohup python DL18/autoaugment/train_cifar.py \
	--model_name=wrn \
	--checkpoint_dir=../../training/ \
	--data_path=../../data/ \
	--dataset='cifar10' \
	--use_cpu=0 \
	> train_cifar_output.txt &
```
