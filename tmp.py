#%%
import numpy as np
import Cycler

#%%

OPdf = Cycler.get_overpotential('220203_NMC','03')


#%%

x=np.array([i for i in range(10)])

print(x)

index = np.array([False] * len(x))
index[3] = True

index2 = np.array([False] * len(x))
index2[1] = True

index3 = index | index2

x[index3] = -1

print(x)

#%%




