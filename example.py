"""
author: Prabhu
email: prabhu.appalapuri@gmail.com
"""
import os
import pandas as pd

# data = []
# path = '/home/prabhu/LSTM/src/.data/imdb/aclImdb/train/neg'
# files = [f for f in os.listdir(path) if os.path.isfile(f)]
# for f in files:
#   with open (f, "r") as myfile:
#     data.append(myfile.read())
#
# df = pd.DataFrame(data)
#
# print(df)



# data = pd.DataFrame()
# name = ['text']
#
# for f in gg('/home/prabhu/LSTM/src/.data/imdb/aclImdb/train/neg/*.txt'):
#     tmp = pd.read_csv(f)
#     data = data.append(tmp)
#
# print(data)
# import csv
# list_ = []
# for f in gg('/home/prabhu/LSTM/src/data/imdb/aclImdb/train/neg/*.txt'):
#     with open(f, 'r') as f:
#         reader = csv.reader(f)
#         list_.append(reader)
#
#
#
# import pandas as pd
# from glob import glob as gg
#
#
# list_data = []
# for file in gg('//home/prabhu/LSTM/src/data/imdb/aclImdb/test/pos/*.txt'):
#     with open(file, 'r') as f:
#         data = f.readline()
#         # print(data)
#         list_data.append(data)
#
# df2 = pd.DataFrame()
# df2['text'] = list_data
# df2.insert(0,'label', 1)

# df3 = pd.concat([df1, df2])
# print(len(df3))
# #
# import pandas as pd
# df = pd.read_csv('/home/prabhu/LSTM/Data/train.csv')
# # pattt = r"(<br\s*[\/]?>)+"
#
# # df.ix[:,1] = df.ix[:,1].replace(pattt, ' ', regex = True)
# #df.to_csv('/home/prabhu/LSTM/Data/test.csv', index = False, header = False)
# # print(df)
# max_n = df.ix[:,1].map(lambda x:len(x)).max()
# # print(df.ix[:,1].apply(lambda x: len(x) !=num))
# # #length_ = max(df.astype('str').applymap(lambda x: len(x)).max())
# # # print(length_)
# min_n = df.ix[:,1].apply(str).map(len).min()
#
# # def f1(s):
# #     return  max(s, key=len())
# #
# # df.groupby(df.ix[:,0]).agg({'df.ix[:,1]': f1})
# # print(df.ix[:,1].str.len())
#
# df_s = df.ix[:,1].str.len().sort_values()
#
# df_d = df[df.ix[:,1].str.len() > 100]
# df_dn = df_d.ix[:,1].map(lambda x:len(x)).min()
#
# df_d.to_csv('/home/prabhu/LSTM/Data/test.csv', index = False, header = False)


import string

all_l = string.ascii_letters

