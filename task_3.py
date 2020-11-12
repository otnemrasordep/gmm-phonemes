import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']

phoneme_1_2 = np.zeros((np.sum(phoneme_id==1) + np.sum(phoneme_id==2), 1)) 
j = 0 
for i in range(len(phoneme_id)):
    if phoneme_id[i] == 1:
        phoneme_1_2[j] = 1
        j += 1
    elif phoneme_id[i] == 2:
        phoneme_1_2[j] = 2 
        j += 1   
            
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
X_phonemes_1_2 = np.zeros((np.sum(phoneme_id==1) + np.sum(phoneme_id==2), 2))
j = 0
for i in range(len(phoneme_id)):
    if phoneme_id[i] == 1 or phoneme_id[i] == 2:
        X_phonemes_1_2[j,:] = X_full[i,:] 
        j += 1
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
GMM_params_phoneme_01_k_03 = np.ndarray.tolist(np.load('data/GMM_params_phoneme_01_k_03.npy', allow_pickle=True))
GMM_params_phoneme_01_k_06 = np.ndarray.tolist(np.load('data/GMM_params_phoneme_01_k_06.npy', allow_pickle=True))
GMM_params_phoneme_02_k_03 = np.ndarray.tolist(np.load('data/GMM_params_phoneme_02_k_03.npy', allow_pickle=True))
GMM_params_phoneme_02_k_06 = np.ndarray.tolist(np.load('data/GMM_params_phoneme_02_k_06.npy', allow_pickle=True))

pred_ph1_k3 = get_predictions(GMM_params_phoneme_01_k_03['mu'], GMM_params_phoneme_01_k_03['s'], GMM_params_phoneme_01_k_03['p'], X_phonemes_1_2)
pred_ph1_k3 = np.sum(pred_ph1_k3, axis=1)
pred_ph2_k3 = get_predictions(GMM_params_phoneme_02_k_03['mu'], GMM_params_phoneme_02_k_03['s'], GMM_params_phoneme_02_k_03['p'], X_phonemes_1_2)
pred_ph2_k3 = np.sum(pred_ph2_k3, axis=1)

pred_ph1_k6 = get_predictions(GMM_params_phoneme_01_k_06['mu'], GMM_params_phoneme_01_k_06['s'], GMM_params_phoneme_01_k_06['p'], X_phonemes_1_2)
pred_ph1_k6 = np.sum(pred_ph1_k6, axis=1)
pred_ph2_k6 = get_predictions(GMM_params_phoneme_02_k_06['mu'], GMM_params_phoneme_02_k_06['s'], GMM_params_phoneme_02_k_06['p'], X_phonemes_1_2)
pred_ph2_k6 = np.sum(pred_ph2_k6, axis=1)

class_k3 = np.zeros((pred_ph1_k3.shape[0],1))
for i in range(len(pred_ph1_k3)):
    if pred_ph1_k3[i] > pred_ph2_k3[i]:
        class_k3[i] = 1
    else:
        class_k3[i] = 2

class_k6 = np.zeros((pred_ph1_k6.shape[0],1))
for i in range(len(pred_ph1_k6)):
    if pred_ph1_k6[i] > pred_ph2_k6[i]:
        class_k6[i] = 1
    else:
        class_k6[i] = 2

n_missclass_k3 = 0
for i in range(len(class_k3)):
    if class_k3[i] != phoneme_1_2[i]:
        n_missclass_k3 += 1

n_missclass_k6 = 0
for i in range(len(class_k6)):
    if class_k6[i] != phoneme_1_2[i]:
        n_missclass_k6 += 1

accuracy_k3 = (1 - (n_missclass_k3 / len(pred_ph1_k3))) * 100
accuracy_k6 = (1 - (n_missclass_k6 / len(pred_ph1_k3))) * 100

########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy_k3))
print('Accuracy using GMMs with 6 components: {:.2f}%'.format(accuracy_k6))
################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()