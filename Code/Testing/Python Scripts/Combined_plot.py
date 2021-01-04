import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extractor(shape,paths, i, attention=False):
    appended_data = []
    errors = dict()
    for p in paths:
        flow = p.replace('\\', '/')
        df_temp = pd.read_excel(flow)
        df_temp = df_temp.iloc[-shape:,:]
        df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
        actual = df_temp[['Actual']]

        if attention:
            df_temp = df_temp.iloc[:, 1]
        else:
            df_temp = df_temp.iloc[:, 0]

        appended_data.append(df_temp)
    df = pd.concat(appended_data, axis=1)
    df = pd.concat([df, actual], axis=1)
    df['Average'] = (df.iloc[:, i].sum(axis=1)) / len(i)
    df['Max'] = df.iloc[:, i].max(axis=1)
    df['Min'] = df.iloc[:, i].min(axis=1)

    return df, final

def shape_finder(path):
    f = path[-1]
    flow = f.replace('\\', '/')
    df_temp = pd.read_excel(flow)
    return df_temp.shape[0]

def combo_plot(df):
    df=df.reset_index()
    df.line.plot()



def files(models):
    models =models+['Ensemble']
    paths =[]
    for i in range(len(models)):
        paths.append(os.getcwd()+ '\\'+models[i]+'\\Forecasting Values.xls')

    shape = shape_finder(paths)
    dfs=[]

    for att in [True,False]:
        print(f'Working on the case when attention is {att}')
        dataframe=extractor(shape,paths,[i for i in range(len(models))],att)
        dfs.append(dataframe)
        plots(dataframe)