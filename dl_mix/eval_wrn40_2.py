import train_cifar

def train_wrn40_2(args):
	x1 = args[0]
	x2 = args[1]
	x3 = args[2]
	a = 3
	b = 100
	acc = (a - x1)**2 + b*(x2 - x1**2)**2+x3**2

	test = train_cifar.main()
	print(test)

	return acc