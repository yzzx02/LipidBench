import pandas as pd

def load_xcms_results(file_path):
    data=pd.read_csv(file_path,index_col=0,sep="\t")
    new_data={}
    sample_cols=[col for col in data.columns if col.endswith(".mzML") ]
    for col in sample_cols:
        data[col]=data[col].fillna(0)
    for index,row in data.iterrows():
        new_data[index]={
            'mz':row['mzmed'],
            'RT':row['rtmed']/60,
        }
        for col in sample_cols:
            new_data[index][col]=row[col]
    pd = pd.DataFrame.from_dict(new_data,orient='index')
    return pd