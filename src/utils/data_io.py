import pandas as pd

def load_xcms_results(file_path):
    data=pd.read_csv(file_path,index_col=0, sep="\t")
    print(data.head())
    