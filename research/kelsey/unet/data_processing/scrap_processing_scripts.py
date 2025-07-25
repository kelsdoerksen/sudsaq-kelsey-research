import numpy as np
import matplotlib.pyplot as plt

def plot_pred_label_resid(arrays):
    combined_data = np.array(arrays)
    #Get the min and max of all your data
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    fig = plt.figure()
    for i in range(len(arrays)):
        ax = fig.add_subplot(len(arrays), 1, i+1)
        #Add the vmin and vmax arguments to set the color scale
        ax.imshow(arrays[i],cmap=plt.cm.YlGn, vmin = _min, vmax = _max)

        ax.autoscale(False)

    plt.show()

ex_prediction = np.load('/Users/kelseyd/Desktop/example_prediction.npy')
ex_label = np.load('/Users/kelseyd/Desktop/example_label.npy')

ex_pred = ex_prediction[0,:,:]
ex_label = ex_label[0,:,:]
residuals = ex_label - ex_pred

combined = np.array([ex_pred,ex_label,residuals])
_min, _max = np.amin(combined), np.amax(combined)


# ---- Plotting ----
fig = plt.figure()
plt.imshow(ex_label)
c = plt.colorbar()
plt.clim(_min, _max)
plt.title('Label')
plt.show()




#plot_pred_label_resid([ex_pred, ex_label, residuals])