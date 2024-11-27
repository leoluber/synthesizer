#%%

import numpy as np
import matplotlib.pyplot as plt
from GaussianProcess import *


"""
    Visualizes the confidence of the Gaussian Process regression
    for different data set sizes
"""
#%%

# # generate some data
# #np.random.seed(0)
# X = np.random.rand(10, 1) * 3 - 1.5
# y = np.sin(X).ravel()
# dy = 0
# y += np.random.normal(0, dy, y.shape)

# # create a Gaussian Process model
# X = np.array(X)
# y = np.array(y)


# #create figure and layout
# plt.figure(figsize=(3, 3))
# # set text size and remove ticks
# plt.rcParams.update({'font.size': 15})
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


# """ loop over different data set sizes """

# def main_loop(X, y):

#     # new data point
#     new_x = np.random.rand(1, 1) * 2 - 1.5

#     for i in range(5):

#         # append new data point
#         X = np.append(X, new_x, axis=0)
#         y = np.append(y, np.sin(new_x).ravel())


#         # fit the model
#         gp = GaussianProcess(training_data=X, targets=y, kernel_type="EXP", model_type="GPRegression")
#         gp.train()

#         # create a test set
#         X_ = np.linspace(-1.5, 1.5, 100)

#         # predict the mean and variance
#         y_pred = np.array([gp.predict([x])[0][0][0] for x in X_])
#         sigma = np.array([gp.predict([x])[1][0][0] for x in X_])

#         # new sample where sigma is max
#         max_sigma = np.argmax(sigma)
#         new_x = X_[max_sigma]

#         # plot the results
#         #plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10,)
#         #plt.plot(X_, y_pred, 'k--')
#         plt.fill_between(X_[:], y_pred - 2* sigma, y_pred + 2* sigma, alpha=0.5, color='grey')
#         plt.scatter(X.ravel(), y, color='k', s=70)
        
#         plt.show()


# main_loop(X, y)

#%%


# def exponential_kernel(x, y, s, l):
exp_kernel = lambda x, y, s, l: s**2 * np.exp(- (1/l**2) * abs(x-y)**2)

# plot the kernel for different values of l
x = np.linspace(-1, 1, 100)
l_values = [5, 2, 1, 0.5, 0.1]
s = 1

plt.figure(figsize=(3,3))
plt.rcParams.update({'font.size': 15})
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

# color according to l (grey scale)
colormap = plt.cm.get_cmap('Greys_r')
colors = [colormap(i) for i in np.linspace(0, 0.7, len(l_values))]
for i, l in enumerate(l_values):
    y = [exp_kernel(x_, 0, s, l) for x_ in x]
    plt.plot(x, y, color=colors[i], linewidth=2, label=f'l = {l}')

# background black
plt.gca().set_facecolor('black')

# x and y labels
plt.xlabel('x', fontweight='bold')
plt.ylabel('k(x, 0)', fontweight='bold')

plt.show()
# %%
