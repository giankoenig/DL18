from hyperopt import hp

def SearchSpace():
  sspace = hp.choice('policies', [
	{'sub_policy': 'Invert_Contrast',
		'Prob1': hp.uniform('Prob_Invert1', 0, 1),
		'Mag1': hp.randint('Mag_Invert1', 1),
		'Prob2': hp.uniform('Prob_Constrast2', 0, 1),
		'Mag2': hp.uniform('Mag_Constrast2', 0.1, 1.9),},
	{'sub_policy': 'Rotate_TranslateX',
		'Prob1': hp.uniform('Prob_Rotate3', 0, 1),
		'Mag1': hp.uniform('Mag_Rotate3', -30, 30),
		'Prob2': hp.uniform('Prob_TranslateX4', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateX44', -150,150),},
	{'sub_policy': 'Sharpness_Sharpness',
		'Prob1': hp.uniform('Prob_Sharpness5', 0, 1),
		'Mag1': hp.uniform('Mag_Sharpness5', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Sharpness6', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness6', 0.1, 1.9),},
	{'sub_policy': 'ShearY_TranslateY',
		'Prob1': hp.uniform('Prob_Shear7', 0, 1),
		'Mag1': hp.uniform('Mag_ShearY7', -0.3, 0.3),
		'Prob2': hp.uniform('Prob_Translate8', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY8', -150,150),},
	{'sub_policy': 'AutoContrast_Equalize',
		'Prob1': hp.uniform('Prob_AutoContrast9', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast9', 1),
		'Prob2': hp.uniform('Prob_Equalize10', 0, 1),
		'Mag2': hp.randint('Mag_Equalize10', 1),},
	{'sub_policy': 'ShearY_TranslateY',
		'Prob1': hp.uniform('Prob_ShearY11', 0, 1),
		'Mag1': hp.uniform('Mag_ShearY11', -0.3, 0.3),
		'Prob2': hp.uniform('Prob_TranslateY12', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY12', -150,150),},
	{'sub_policy': 'Color_Brightness',
		'Prob1': hp.uniform('Prob_Color13', 0, 1),
		'Mag1': hp.uniform('Mag_Color13', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Brightness14', 0, 1),
		'Mag2': hp.uniform('Mag_Brightness14', .1, 1.9),},
	{'sub_policy': 'Sharpness_Brightness',
		'Prob1': hp.uniform('Prob_Sharpness15', 0, 1),
		'Mag1': hp.uniform('Mag_Sharpness15', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Brightness16', 0, 1),
		'Mag2': hp.uniform('Mag_Brightness16', .1, 1.9),},
	{'sub_policy': 'Equalize_Equalize',
		'Prob1': hp.uniform('Prob_Equalize17', 0, 1),
		'Mag1': hp.randint('Mag_Equalize17', 1),
		'Prob2': hp.uniform('Prob_Equalize18', 0, 1),
		'Mag2': hp.randint('Mag_Equalize18', 1),},
	{'sub_policy': 'Contrast_Sharpness',
		'Prob1': hp.uniform('Prob_Constrast19', 0, 1),
		'Mag1': hp.uniform('Mag_Constrast19', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Sharpness20', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness20', 0.1, 1.9),},
	{'sub_policy': 'Color_TranslateX',
		'Prob1': hp.uniform('Prob_Color21', 0, 1),
		'Mag1': hp.uniform('Mag_Color21', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_TranslateX22', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateX22', -150,150),},
	{'sub_policy': 'Equalize_AutoContrast',
		'Prob1': hp.uniform('Prob_Equalize23', 0, 1),
		'Mag1': hp.randint('Mag_Equalize23', 1),
		'Prob2': hp.uniform('Prob_AutoContrast24', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast24', 1),},
	{'sub_policy': 'TranslateY_Sharpness',
		'Prob1': hp.uniform('Prob_TranslateY25', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY25', -150,150),
		'Prob2': hp.uniform('Prob_Sharpness26', 0, 1),
		'Mag2': hp.uniform('Mag_Sharpness26', 0.1, 1.9),},
	{'sub_policy': 'Brightness_Color',
		'Prob1': hp.uniform('Prob_Brightness27', 0, 1),
		'Mag1': hp.uniform('Mag_Brightness27', .1, 1.9),
		'Prob2': hp.uniform('Prob_Color28', 0, 1),
		'Mag2': hp.uniform('Mag_Color28', 0.1, 1.9),},
	{'sub_policy': 'Solarize_Invert',
		'Prob1': hp.uniform('Prob_Polarize29', 0, 1),
		'Mag1': hp.uniform('Mag_Polarize29', 4, 8),
		'Prob2': hp.uniform('Prob_Invert30', 0, 1),
		'Mag2': hp.randint('Mag_Invert30', 1),},
	{'sub_policy': 'Equalize_AutoContrast',
		'Prob1': hp.uniform('Prob_Equalize31', 0, 1),
		'Mag1': hp.randint('Mag_Equalize31', 1),
		'Prob2': hp.uniform('Prob_AutoContrast32', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast32', 1),},
	{'sub_policy': 'Equalize_Equalize',
		'Prob1': hp.uniform('Prob_Equalize33', 0, 1),
		'Mag1': hp.randint('Mag_Equalize33', 1),
		'Prob2': hp.uniform('Prob_Equalize34', 0, 1),
		'Mag2': hp.randint('Mag_Equalize34', 1),},
	{'sub_policy': 'Color_Equalize',
		'Prob1': hp.uniform('Prob_Color35', 0, 1),
		'Mag1': hp.uniform('Mag_Color35', 0.1, 1.9),
		'Prob2': hp.uniform('Prob_Equalize36', 0, 1),
		'Mag2': hp.randint('Mag_Equalize36', 1),},
	{'sub_policy': 'AutoContrast_Solarize',
		'Prob1': hp.uniform('Prob_AutoContrast37', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast37', 1),
		'Prob2': hp.uniform('Prob_Polarize38', 0, 1),
		'Mag2': hp.uniform('Mag_Polarize38', 4, 8),},
	{'sub_policy': 'Brightness_Color',
		'Prob1': hp.uniform('Prob_Brightness39', 0, 1),
		'Mag1': hp.uniform('Mag_Brightness39', .1, 1.9),
		'Prob2': hp.uniform('Prob_Color40', 0, 1),
		'Mag2': hp.uniform('Mag_Color40', 0.1, 1.9),},
	{'sub_policy': 'Solarize_AutoContrast',
		'Prob1': hp.uniform('Prob_Polarize41', 0, 1),
		'Mag1': hp.uniform('Mag_Polarize41', 4, 8),
		'Prob2': hp.uniform('Prob_AutoContrast42', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast42', 1),},
	{'sub_policy': 'TranslateY_TranslateY',
		'Prob1': hp.uniform('Prob_TranslateY43', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY143', -150,150),
		'Prob2': hp.uniform('Prob_TranslateY44', 0, 1),
		'Mag2': hp.uniform('Mag_TranslateY44', -150,150),},
	{'sub_policy': 'AutoContrast_Solarize',
		'Prob1': hp.uniform('Prob_AutoContrast45', 0, 1),
		'Mag1': hp.randint('Mag_AutoContrast45', 1),
		'Prob2': hp.uniform('Prob_Polarize46', 0, 1),
		'Mag2': hp.uniform('Mag_Polarize46', 4, 8),},
	{'sub_policy': 'Equalize_Invert',
		'Prob1': hp.uniform('Prob_Equalize47', 0, 1),
		'Mag1': hp.randint('Mag_Equalize47', 1),
		'Prob2': hp.uniform('Prob_Invert48', 0, 1),
		'Mag2': hp.randint('Mag_Invert48', 1),},
	{'sub_policy': 'TranslateY_AutoContrast',
		'Prob1': hp.uniform('Prob_TranslateY49', 0, 1),
		'Mag1': hp.uniform('Mag_TranslateY49', -150,150),
		'Prob2': hp.uniform('Prob_AutoContrast50', 0, 1),
		'Mag2': hp.randint('Mag_AutoContrast50', 1),},])

  return sspace