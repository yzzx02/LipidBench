import pandas as pd

def load_xcms_results(file_path):
    data=pd.read_csv(file_path,index_col=0)
    new_data={}
    sample_cols=[col for col in data.columns if col.endswith(".mzML") ]
    for col in sample_cols:
        data[col]=data[col].fillna(0)
    for index,row in data.iterrows():
        new_data[index]={
            'mz':round(row['mzmed'],4),
            'RT':round(row['rtmed']/60,3),
            'npeaks':row['npeaks'],
            
        }
        for col in sample_cols:
            new_data[index][col]=row[col]
    pd = pd.DataFrame.from_dict(new_data,orient='index')
    pd.to_csv(file_path,index_label='Feature_id',float_format='%.4f')
    return pd

def load_msdial_results(file_path,outputfile):
    data=pd.read_excel(file_path,index_col=0)
    sample_cols=[col for col in data.columns if col.endswith(".mzML") ]
    data[sample_cols] = data[sample_cols].fillna(0)
    column_map = {
            'Precursor m/z': 'mz',
            'RT left(min)': 'RTmin',
            'RT (min)': 'RT',
            'RT right (min)': 'RTmax',
            'Height': 'Height',
            'Area': 'Area',
            'Estimated noise': 'Estimated noise',
            'S/N': 'S/N',
            'Sharpness': 'Sharpness',
            'Gaussian similarity': 'Gaussian similarity',
            'Ideal slope': 'Ieal slope',
            'Symmetry': 'Symmetry'
        }
    keep_cols = list(column_map.keys()) + sample_cols
    df = data[keep_cols].copy()
    df.rename(columns=column_map, inplace=True)
    df.to_csv(f'{outputfile}', index=True, index_label='Feature_ID')
    print(f"Processed MS-DIAL results saved to {outputfile}")
    #添加一列数字标记feature id
    return df

def load_pyopenms_results(file_path):
    df=pd.read_csv(file_path,index_col=0)
    # 删除无关列：sequence(序列), charge(电荷), quality(质量评分)
    cols_to_drop = ['sequence', 'charge', 'quality']
    df = df.drop(columns=[c for c in df.columns if c in cols_to_drop], errors='ignore')

    # 对RT列除以60,3位小数，mz列保留4位小数
    if 'mz' in df.columns:
        df['mz'] = df['mz'].round(4)
    if 'RT' in df.columns:
        df['RT'] = (df['RT'] / 60.0).round(3)

    if 'Feature_id' not in df.columns:
        df.insert(0, 'Feature_id', range(1, len(df) + 1))
    file_path = file_path.parent / f"{file_path.stem}_processed.csv"
    # 保存 CSV，float_format='%.4f' 可以避免科学计数法
    df.to_csv(file_path, index=True, index_label='Feature_ID', float_format='%.4f')
    print(f"Processed pyOpenMS results saved to {file_path}")
    return df