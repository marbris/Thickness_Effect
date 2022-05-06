
#%%
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import matplotlib.pyplot as plt
import numpy as np

filename = 'Data/McMullin/EIS Symmetric cells/0405_Data.DTA'
frequencies, Z = preprocessing.readGamry(filename)


circuit = 'CPE0-R0'
initial_guess = [0.05, .8, 50]
circuit = CustomCircuit(circuit, initial_guess=initial_guess)


circuit.fit(frequencies, Z)

Z_fit = circuit.predict(frequencies)

print(circuit)

#%%

fig, ax = plt.subplots(figsize=(6,13))

ax = circuit.plot(f_data=frequencies, Z_data=Z, kind='nyquist', ax=ax)

#ax.plot(np.real(Z), -np.imag(Z), '.')
#ax.plot(np.real(Z_fit), -np.imag(Z_fit), '-')

fig.tight_layout()
plt.show()


#%%


plot_nyquist(ax, Z, fmt='o')
plot_nyquist(ax, Z_fit, fmt='-')

plt.legend(['Data', 'Fit'])


plt.show()