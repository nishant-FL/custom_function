#!/usr/bin/env python
# coding: utf-8

# In[138]:


# Linear Regression
def create_model(model_name, modeltype):
    """
    create regression
    modeltype:
        "lr"`'LinearRegression'
        "ridge"`'Ridge'
        "kmeans"`clustering
        "ann"`'artifical neural network'
        
    """
    step1 = f"# model import for {modeltype}"
    step2 = f"# model initalize for {modeltype}"
    step3 = f"# tunable parameters for {modeltype}"
    step4 = f"# selecting the optimal parameter for {modeltype} "
    step5 = f"# fit {modeltype} on training data" 
    step6 = f"# predict using {modeltype} on test data"
    step6 = f"# Evaluate {modeltype}: {model_name}"

    if modeltype == "lr":
        print(step1)
        model_import = "from sklearn.linear_model import LinearRegression"
        print(model_import + "\n")

        print(step2)
        model_initalise = f"{model_name} = LinearRegression()"
        print(model_initalise + "\n")

        print(step3)
        model_tune = "None"
        print(model_tune + "\n")

    if modeltype == "ridge":
        print(step1)
        model_import = "from sklearn import linear_model"
        print(model_import + "\n")
        
        print(step2)
        model_initalise = f"{model_name} = linear_model.Ridge(alpha=.5)"
        print(model_initalise + "\n")
        
        print(step3)
        model_tune = "alpha"
        print(model_tune + "\n")

    if modeltype == "kmeans":
        print(step1)
        model_import = "from sklearn.cluster import KMeans"
        print(model_import + "\n")
        
        print(step2)
        model_initalise = f"{model_name} = KMeans(n_clusters=2, random_state=42)"
        print(model_initalise + "\n")
        
        print(step3)
        model_tune = "n_clusters"
        print(model_tune + "\n")

    if modeltype == "ann":
        variables = input("number of dimensions: ")
        
        print(step1)
        model_import_1 = "from keras.models import Sequential"
        model_import_2 = "from keras.layers import Dense, Activation"
        print(model_import_1)
        print(model_import_2)
        
         
        step2 = f"# model initalize for {modeltype}"
        
        
        
        
        step3 = f"# tunable parameters for {modeltype}"
        step4 = f"# selecting the optimal parameter for {modeltype} "
        step5 = f"# fit {modeltype} on training data" 
        step6 = f"# predict using {modeltype} on test data"
        step6 = f"# Evaluate {modeltype}: {model_name}"
        
        


# In[ ]:


#create_model("nn","ann")


# In[16]:


def preprocess_task(task):
    """
    task:
    `train_test_split
    `standardize
    `dtypes_change
    `one-hot
    `
    """
    step1 = f"# module import  for {task}"
    step2 = f"# module initalize for {task}"
    step3 = f"# check for"

    if task == "train_test_split":
        test_size = input("test_size: ") + " \n"
        stratify = input("stratify = None , y: ")
        
        print(step1 + f" with test size = {test_size}")
        module_import= "from sklearn.model_selection import train_test_split"
        print(module_import + "\n")
        
        print("# Dataframe of Independent Variables ")
        print("X =... " + "\n")
        
        print("# Data of Target Variable")
        print("y =..." + "\n")
        
        print("# Splitting the dataset into the Training set and Test set")
        model_initalise = f"X_train, X_test, y_train,y_test = train_test_split(X,y,test_size={test_size},stratify={stratify}, random_state = 42)"
        print(model_initalise)
        
        print("print('train_test_split_shape: ','X_train:',X_train.shape,'y_train:', y_train.shape,'X_test:', X_test.shape,'y_test:', y_test.shape)")
            
    if task == "standardize":
        scaler = input("create scaler as:")
        data = input("create numeric_dataframe as:")
        scaled_data = input("create scaled_data as:")
        
        print (step1)
        module_import = "from sklearn.preprocessing import StandardScaler" 
        print (module_import + "\n")
        
        print (step2)
        model_initalise = f"{scaler} = StandardScaler()"
        print (model_initalise + "\n")
        
        step3 = f"# Create object of numeric data to fit {scaler}" 
        print (step3)
        print (f"{data} = ...\n")
        
        step4 = f"# fit {scaler} object on {data}" 
        print(step4)
        print (f"{scaler}.fit({data})\n")
        
        step5 = f"# transform {data} using {scaler} and save to object {scaled_data}"
        print(step5)
        print(f"{scaled_data} = {scaler}.transform({data})")
        
        
            
    if task == "dtypes_change":
        data = input("dataframe name: ")
        col_list = input("column names:")
        dtype = input("'object'/'int64'/'float64'/'bool'/'datetime64'/'timedelta[ns]'/'category'")
        print (f"for col in[{col_list}]",":")
        print (f"\t{data}[col] = {data}[col].astype({dtype})")
        print (f"assert {data}.loc[:,[{col_list}]].dtypes.values.all() == {dtype}")
        


# In[1]:


#preprocess_task("standardize")


# In[ ]:




