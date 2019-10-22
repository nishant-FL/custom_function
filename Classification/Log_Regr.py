class Log_Regr(object):
	def __init__(self):
        print ("You have selected and initialized the Logistic Regression Class\n")


    def log_regr_(self, model_name, modeltype):
    	df_X = input("Your Training DataFrame with only features (X): ")
    	df_Y = input("Your Training DataFrame with only Labels (Y): ")
    	df_t_X = input("Your Testing DataFrame with only features (X): ")
    	print ("--------copy script below------------")
    	print()
    	print("# Import Logistic Libraries")
    	print(f"from sklearn.linear_model import LogisticRegression")
    	print(f"from sklearn.model_selection import GridSearchCV")
    	print()

    	print(f"# Create an instance / initialize the model")
    	print(f"{model_name} = LogisticRegression(solver='lbfgs', n_jobs=2, random_state=351)")
    	print()

    	print(f"# Most important tunable parameters for {modeltype}:")
    	print(f"# C='Default Value is 1.0. Inverse of regularization strength; must be a positive float'")
    	print(f"# penalty='str, l1, l2, elasticnet or none, optional (default=l2)")
    	print(f"# solver='lbfgs, liblinear, saga'")
    	print()

    	print(f"# Function to get tuned model with best parameters")
    	print(f"C=np.logspace(-5,5,10)")
    	print(f"penalty=['l1', 'l2', 'elasticnet']")
    	print(f"solver=['lbfgs', 'liblinear', 'saga']")
    	print(f"multi_class_options = ['ovr', 'multinomial']")
		print(f"class_weight_options = ['None', 'balanced']")
    	print(f"hyperparams = dict(C=C, penalty=penalty, solver=solver, multi_class=multi_class_options, class_weight=class_weight_options)")
    	print()

    	print(f"# Instantiating the GridSearchCV object:")
		print(f"clf = GridSearchCV({model_name}, hyperparams, cv = 5, verbose=0, scoring='roc_auc') ")
		print()

		print(f"# Find the best best_estimator_:")
		print(f"print('Best Estimator is {}'.format(clf.best_estimator_))")
		print()

		print(f"# View best hyperparameters and scores:")
		print(f"print('Best Penalty:', clf.best_estimator_.get_params()['penalty'])")
		print(f"print('Best C:', clf.best_estimator_.get_params()['C'])")
		print(f"print('Best Solver:', clf.best_estimator_.get_params()['solver'])")
		print(f"print('Best Multi_class_option:', clf.best_estimator_.get_params()['multi_class'])")
		print(f"print('Best Class_weight_option:', clf.best_estimator_.get_params()['class_weight'])")
		print(f"print('Best roc_auc is {}'.format(clf.best_score_))")
		print()

		print(f"# Fit on the best model from GridSearchCV")
		print(f"best_model = clf.fit({df_X},{df_Y})")
		print()

		print(f"# Predict class labels for samples in test(X), using the best model:")
		print(f"predict = best_model.predict({df_t_X})")
		print(f"# Predict probabilities of class labels for samples in test(X), using the best model:")
		print(f"predict_prob = best_model.predict_proba({df_t_X})")

