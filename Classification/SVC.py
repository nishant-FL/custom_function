# This file contains model definition for SVC model:

class SVC(object):
    def svc_(self, model_name, modeltype):
        self.greet = "You have selected and initialized the Support vector Classification\n"
        print (self.greet)

        df_X = input("Your Training DataFrame with only features (X_train): ")
        df_Y = input("Your Training DataFrame with only Labels (Y_train): ")
        df_t_X = input("Your Testing DataFrame with only features (X_test): ")
        df_t_Y = input("Your Testing DataFrame with only Labels (Y_test): ")

        print()
        print ("--------copy script below------------")
        print()
        print("# Import SVC Libraries")
        print(f"from sklearn.svm import SVC")
        print(f"from sklearn.model_selection import GridSearchCV")
        print()
        print(f"# Create an instance / initialize the model")
        print(f"{model_name} = SVC(n_jobs=2, random_state=351)")
        print()

        print(f"# Most important tunable parameters for {modeltype}:")
        print(f"# C = With default value of 1.0, it's the regularization parameter (C) of the error term")
        print(f"# kernel = With default value of 'rbf', it specifies the kernel type to be used. Values can be 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'.")
        print(f"# degree = With default value of 3, it is the degree of the polynomial kernel function ('poly') and is ignored by all other kernels.")
        print(f"# gamma = With default value of 'auto', it is the kernel coefficient for 'rbf', 'poly', and 'sigmoid'. With default value, 1/n_features will be used. If gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma. The current default of gamma, 'auto', will change to 'scale' in version 0.22.")
        print()

        print(f"# Function to get tuned model with best parameters")
        print('''param_grid = {
        	'C': [0.001, 0.01, 0.1, 1, 10, 100 ],
        	'gamma': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        	'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        	'degree':[*range(0,10)]
        	}''')
        print()

        print(f"# Instantiating the GridSearchCV object:")
        print(f"clf = GridSearchCV(estimator = {model_name}, param_grid = param_grid, cv = 10, verbose=0, scoring='roc_auc')")
        print()

        print(f"# Find the best best_estimator_:")
        str="print('Best Estimator is {}'.format(clf.best_estimator_))"
        print(str)
        print()
        print(f"# View best hyperparameters and scores:")
        str="print('Best parameters are {}'.format(clf.best_params_))"
        print(str)
        str="print('Best scores are {}'.format(clf.best_score_))"
        print(str)
        print()

        print(f"# Fit on the best model from GridSearchCV")
        print(f"best_model = clf.fit({df_X},{df_Y})")
        print()
        print(f"# Predict class labels for samples in test(X), using the best model:")
        print(f"predict = best_model.predict({df_t_X})")
        print()
        print(f"# Predict probabilities of class labels for samples in test(X), using the best model:")
        print(f"predict_prob = best_model.predict_proba({df_t_X})")
        print()

        print(f"# Create Confusion matrix and evalute the scores")
        print(f"from sklearn.metrics import confusion_matrix, accuracy_score, f1_score")
        print()
        print(f"# Predict for Confusion_matrix")
        print(f"train_predictions = best_model.predict({df_X})")
        print(f"test_predictions = best_model.predict({df_t_X})")
        print()

        print(f"df_Y = {df_Y}")
        print(f"df_X = {df_X}")
        print(f"df_t_X = {df_t_X}")
        print(f"df_t_Y = {df_t_Y}")
        print()

        print(f"# Train data accuracy")

        str=r"print('TRAIN DATA ACCURACY',accuracy_score({df_Y},train_predictions))"
        print(str)
        str=r"print('\nTrain data f1-score for class '1'',f1_score({df_Y},train_predictions,pos_label=1))"
        print(str)
        str=r"print('\nTrain data f1-score for class '2'',f1_score({df_Y},train_predictions,pos_label=2))"
        print(str)
        print()

        print(f"# Test data accuracy")
        str=r"print('\n\n--------------------------------------\n\n')"
        print(str)
        str=r"print('TEST DATA ACCURACY',accuracy_score({df_t_Y},test_predictions))"
        print(str)
        str=r"print('\nTest data f1-score for class '1'',f1_score({df_t_Y},test_predictions,pos_label=1))"
        print(str)
        str=r"print('\nTest data f1-score for class '2'',f1_score({df_t_Y},test_predictions,pos_label=2))"
        print(str)
        print()

        print(f"# Confusion_matrix for Test Data:")
        print(f"cm = confusion_matrix({df_t_Y},test_predictions)")
        print(f"accuracy = float(cm.diagonal().sum())/len({df_t_Y})")
        str=r"print('\nAccuracy Of SVM For The Given Dataset : ', accuracy)"
        print(str)
