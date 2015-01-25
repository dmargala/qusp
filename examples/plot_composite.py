import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import qusp

stack = h5py.File('composite/sdss-dr9-qso-stack.hdf5')
corrected_stack = h5py.File('composite/sdss-dr9-qso-stack-tpcorr.hdf5')
sdss_stack = h5py.File('sdss-dr7-qso-stack.hdf5')

xlim = [850, 2900]
fontsize = 16

fig = plt.figure(figsize=(16,8))
# plt.plot(dr7[0], dr7[1], c='red', label='SDSS-DR7')
# plt.plot(dr7[0], dr7[2], c='blue', label='SDSS-DR9')

gs1 = gridspec.GridSpec(2, 1)
gs1.update(hspace=0.00)

plt.subplot(gs1[0])

model = (stack['wavelength'].value/1450)**(-1.5)
model[stack['wavelength'].value < 1216] *= np.exp(-.0018*(1+2.65)**3.92)
plt.plot(stack['wavelength'], sdss_stack['median_flux'].value, 
         c='black', label='SDSS')
plt.plot(stack['wavelength'], stack['median_flux'].value, 
         c='red', label='BOSS')
plt.plot(stack['wavelength'], corrected_stack['median_flux'].value, 
         c='blue', label='Corrected BOSS')
#plt.plot(stack['wavelength'],model)
#plt.plot(corrected_stack['wavelength'], corrected_stack['median_flux'], c='blue', label='SDSS-DR9-Corrected')
plt.ylim([-.1,2.2])
plt.xlim(xlim)
plt.legend(prop={'size':fontsize})
plt.tick_params(labelsize=fontsize)
plt.grid()
plt.ylabel(r'Normalized Flux', size=fontsize)

plt.subplot(gs1[1])
# plt.plot(stack['wavelength'], sdss_stack['median_flux'].value, 
#          c='red', label='SDSS')
plt.axhline(0, c='black', label='SDSS/SDSS - 1')
plt.plot(stack['wavelength'], (stack['median_flux'].value-sdss_stack['median_flux'].value)/sdss_stack['median_flux'].value, 
         c='red', label='BOSS/SDSS - 1')
plt.plot(stack['wavelength'], (corrected_stack['median_flux'].value-sdss_stack['median_flux'].value)/sdss_stack['median_flux'].value, 
         c='blue', label='Corrected BOSS/SDSS - 1')
plt.xlim(xlim)
plt.ylim([-.5,+.5])
plt.legend(prop={'size':fontsize})
plt.tick_params(labelsize=fontsize)
plt.grid()
plt.ylabel(r'Normalized Flux Ratio', size=fontsize)
#plt.legend()

plt.xlabel(r'Restframe Wavlength $(\AA)$', size=fontsize)
fig.savefig('composite/repeat-comparison.pdf', bbox_inches='tight')