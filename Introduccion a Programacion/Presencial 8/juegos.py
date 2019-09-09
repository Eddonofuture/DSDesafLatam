#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../Presencial 8'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd

df = pd.read_csv('athlete_events.csv')
df.head()


#%%
ejercicio_1 = df.shape
ejercicio_1


#%%
ejercicio_2 = df["Games"].unique().size
ejercicio_2


#%%
ejercicio_3 = df["Season"].value_counts(normalize = True)
ejercicio_3


#%%
pre_4 = df[df["Season"] == "Summer"]
post_4 = pre_4[pre_4["Year"] == pre_4["Year"].min()]
ejercicio_4 = post_4["City"].tolist()[0]
ejercicio_4


#%%
pre_5 = df[df["Season"] == "Winter"]
post_5 = pre_5[pre_5["Year"] == pre_5["Year"].min()]
ejercicio_5 = post_5["City"].tolist()[0]
ejercicio_5


#%%
pre_6 = df["Team"].value_counts()
ejercicio_6 = pre_6.head(10)
ejercicio_6


#%%
ejercicio_7 = df["Medal"].value_counts(normalize = True, dropna=True)
ejercicio_7


#%%
pre_8 = df[df["Season"] == "Summer"]
post_8 = pre_8[pre_8["Year"] == pre_8["Year"].min()]
ejercicio_8 = post_8["Team"].unique()
ejercicio_8


#%%



