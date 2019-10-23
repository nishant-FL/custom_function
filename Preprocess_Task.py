class Preprocess_Task:
    def __init__(self):
        
        self.get_script = "---copy script below---\n"
        
    def missing_values_chk(self):
        data = input("dataframe: ")
        
        print(self.get_script)
        
        step1 = f"# find cols with sum of missing value in {data}"
        print (step1)
        
        print(f"for col in {data}.columns:")
        print(f"\tif {data}[col].isnull().sum()> 0:")
        print(f"\t print(col,{data}[col].dtype,{data}[col].isnull().sum(),np.round({data}[col].isnull().sum().sum()/{data}.shape[0]*100,2),'%')")
        print(f"print('total missing values: ', np.round({data}.isnull().sum().sum()/{data}.shape[0]*100,2),'%')")
        
    def train_test_split(self):
        test_size = input("test_size: ") + " \n"
        stratify = input("stratify = None , y: ")
        
        print(self.get_script)
        
        step1 = f"# module import  for train_test_split"
        print(step1 + f" with test size = {test_size}")
        module_import= "from sklearn.model_selection import train_test_split"
        print(module_import)
        
        print("# Dataframe of Independent Variables ")
        print("X =... " + "\n")
        
        print("# Data of Target Variable")
        print("y =..." + "\n")
        
        print("# Splitting the dataset into the Training set and Test set")
        model_initalise = f"X_train, X_test, y_train,y_test = train_test_split(X,y,test_size={test_size},stratify={stratify}, random_state = 42)"
        print(model_initalise)
        
        print("print('train_test_split_shape: ','X_train:',X_train.shape,'y_train:', y_train.shape,'X_test:', X_test.shape,'y_test:', y_test.shape)")
        
    def standardize(self):
        scaler = input("create scaler as:")
        data = input("create numeric_dataframe as:")
        scaled_data = input("create scaled_data as:")
        
        print(self.get_script)
        
        step1 = "# module import  for to standardise data"
        print (step1)
        module_import = "from sklearn.preprocessing import StandardScaler" 
        print (module_import + "\n")
        
        step2 = "# module initalization to standardise data "
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
    
    def dtypes_change(self):
        data = input("dataframe name: ")
        col_list = input("column names:")
        dtype = input("'object'/'int64'/'float64'/'bool'/'datetime64'/'timedelta[ns]'/'category'")
        
        print(self.get_script)
        
        step1 = f"# chage data type of {col_list} to {dtype}"
        print (step1)
        
        print (f"for col in[{col_list}]",":")
        print (f"\t{data}[col] = {data}[col].astype({dtype})")
        print (f"assert {data}.loc[:,[{col_list}]].dtypes.values.all() == {dtype}")
        
    def explore(self):
        data = input("dtaframe name: ")
        top = input ("top n rows: ")
        last = input ("bottom n rows: ")
        
        
        print(self.get_script)
        
        step1 = f"# column names in {data}"
        print (step1)
        print (f"{data}.columns\n")
        
        step2 = f"# top {top} rows of {data}"
        print(step2)
        print(f"{data}.head({top})\n")
        
        step3 = f"# top {last} rows of {data}"
        print(step3)
        print(f"{data}.tail({top})\n")
        
        step4 = f"# data type of colums in {data}"
        print(step4)
        print (f"{data}.info()\n")
        
        step5 = f"# colunwise unique values in {data}"
        print(step5)
        print(f"{data}.nunique()\n")
        
        step6 = f"# summary statistic for {data}"
        print(step6)
        print(f"{data}.describe(percentiles = [.5,.25, .5, .75,.95],include= 'all')")
        
        
