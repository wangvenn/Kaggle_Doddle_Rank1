import pandas as pd
import numpy as np
import os

EPS = 1e-6
simp_path = 'D:/Users/ad_yan/Desktop/yad/8Doodle/data/train_simplified/'
raw_path = 'E:/train_raw/'
output_path = 'D:/Users/ad_yan/Desktop/yad/8Doodle/data/train_simplified_time/'

files = os.listdir(simp_path)
if len(files)!=340:
    raise Warning('Check your data folder')
    
def do_one(param):

    df_r, df_s, n = param
    assert(df_r.key_id[n] == df_s.key_id[n])
    print('\r\t%d  %s'%(n, df_r.key_id[n]), end ='', flush=True)

    drawing_s = eval(df_s.drawing[n])
    drawing_r = eval(df_r.drawing[n])
    assert(len(drawing_s)==len(drawing_r))

    drawing=[]
    for i, (d_s,d_r) in enumerate(zip(drawing_s,drawing_r)):
        x_s,y_s     = np.array(d_s)
        x_r,y_r,t_r = np.array(d_r)

        N_s = len(x_s)
        N_r = len(x_r)
        if (N_s>1) and (N_r>1):
            d_s = ((x_s[1:]-x_s[:-1])**2 + (y_s[1:]-y_s[:-1])**2 )**0.5
            d_s = np.insert(d_s, 0, 0)
            distance = d_s.sum()+ EPS
            t_s = (d_s.cumsum()/distance)*(t_r[-1]-t_r[0]) + t_r[0]

        elif (N_s==1) and (N_r==1):
            t_s = t_r
        elif (N_s==1) and (N_r>1):
            t_s = t_r[[0]]
        elif (N_s>1) and (N_r==1):
            t_s = t_r[[0]*N_s]

        t_s = list(t_s.astype(np.int32))
        x_s = list(x_s)
        y_s = list(y_s)
        drawing.append([x_s,y_s,t_s])
    drawing= str(drawing)
    return drawing


for filename in files:
    df_r = pd.read_csv(raw_path +  filename)
    df_s = pd.read_csv(simp_path + filename)
    df_s_time = df_s.copy()
    for i in range(df_s_time.shape[0]):
        df_s_time['drawing'][i] = do_one((df_r, df_s, i))
    df_s_time.to_csv(output_path + filename, index=False)
    print(filename, ' done!')
    del df_s_time
