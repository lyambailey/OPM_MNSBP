# The purpose of this script is to compute SEF, mu, and beta signal-to-noise ratios (SNRs) for sensor and label data

## Import packages
import os
import mne
import numpy as np
import pandas as pd

################################################
## User input

# Task prefix
dataPre = 'MNS_shortISI'

# Subject list
subjects = [1,2,3,4,5,6,7,8,9,10,11,12]  # check this does not includes bads

# Write data?
writeOK = True

# Define time window for N20m 
N20m_window = [0.021, 0.023] # 3 ms peak identified in 10_SEF_stats.r
################################################
## Other params

# Define number of subjects
if dataPre == 'buttonPress':
	n=11
else:
	n=12

# Define labels (labels) 
labels = ['S_precentral-sup-part-lh', 'G_precentral-lh',
		'S_central-lh', 'G_postcentral-lh',
		'S_postcentral-lh', 'G_parietal_sup-lh']

method_sufx = 'MNE-constrained'

# Define empty list to store individulal subject data
sens_evokeds = []
bad_chs = []

################################################
## Define file paths 

# Main directories
projectDir = '../'
resultsDir = os.path.join(projectDir, 'results')
procDir = os.path.join(projectDir, 'proc_data')
supportDir = os.path.join(projectDir, 'support_files')

# Input files (sensor data for individual subjects)
evoked_fstem = '-trans-cleaned-manRej-baseCorrected-evoked.fif'   	# evoked
tfrs_fstem = '-trans-cleaned-manRej-baseCorrected-tfrs.fif'  # baseline-corrected tfrs 

# Input files (label data for all subjects)
allSubjs_labelEvoked_fname = os.path.join(resultsDir, 
									f'{dataPre}-trans-cleaned-manRej-baseCorrected-{method_sufx}-labelEvoked_N={n}.npy') # evoked 
allSubjs_labelTFRcorr_fname = os.path.join(resultsDir, 
										f'{dataPre}-trans-cleaned-manRej-{method_sufx}-baseCorrected-labelTFRs_N={n}.npy') # baseline-corrected tfrs

# Other
epoch_params_fname = os.path.join(supportDir, 'all_tasks_epoch_durations.txt') # epoch timing parameters
times_evoked_fname = os.path.join(resultsDir, f'MNS_shortISI-sens-evoked-times.npy')


################################################
## Define functions

# Function to read epoch parameters
def read_timing_params(epochParamsFile):
	
	# Define empty dict to store timings
	epochParams = {}

	# Read the file line by line
	with open(epochParamsFile, "r") as file:
		for line in file:
			# Skip lines that start with '#' or are empty
			if line.strip() and not line.startswith("#"):
				# Split the line at '='
				key, value = line.split("=", 1)
				key = key.strip()  # Remove whitespace around the key
				value = float(value.strip())  # Convert the value to float
				epochParams[key] = value  # Store in the dictionary

	return epochParams

# Function to read sensor-level evoked data for a single subject
def read_evoked_data(subjectID, evoked_fstem):
	
	# Define path to data
	subDir = os.path.join(procDir, subjectID, 'meg')
	evoked_fname = os.path.join(subDir, f'{dataPre}{evoked_fstem}')
	
	# Read the evoked data
	evoked = mne.read_evokeds(evoked_fname)
	
	return evoked

# Function to read sensor-level TFR data for a single subject
def read_tfr_data(subjectID, tfrs_fstem):
	# Define path to data
	subDir = os.path.join(procDir, subjectID, 'meg')
	tfrs_fname = os.path.join(subDir, f'{dataPre}{tfrs_fstem}')
	
	# Read the TFR data
	tfrs = mne.time_frequency.read_tfrs(tfrs_fname) # Note, this is a "TFR from evoked" object because of the way it was created. The data was NOT computed from evoked data. 
	
	return tfrs

# Function to compute SNR for a single sensor / label
def compute_snr(data, times, signal_window, noise_window):   # Data should have shape (subjects, channels, timepoints)

    # Get indices for signal and noise windows
    signal_idxs = np.where((times >= signal_window[0]) & (times <= signal_window[1]))[0]
    noise_idxs = np.where((times >= noise_window[0]) & (times <= noise_window[1]))[0]
    
    # Compute mean over the signal and noise windows
    signal_mean = np.mean(data[:, :, signal_idxs], axis=2)
    noise_mean = np.mean(data[:, :, noise_idxs], axis=2)

    noise_sd = np.std(data[:, :, noise_idxs], axis=2)

    # Compute SNR as the difference between the means divided by the standard deviation
    snr = np.abs(signal_mean - noise_mean) / noise_sd

    return snr

# Function to return channel with best SNR for each subject
def get_best_channel(snr, channels):  

	# Extract max SNR (across channels) for each subject
	max_snrs = np.max(snr, axis=1)
	max_snr_chs = [channels[idx] for idx in np.argmax(snr, axis=1).tolist()]

	return max_snrs, max_snr_chs
	

################################################
## Load data

# Read times
times = np.load(times_evoked_fname, allow_pickle=True)

# Read epoch parameters
epochParams = read_timing_params(epoch_params_fname)
baselineStart = epochParams[f'{dataPre}_baselineStart']  # Start of baseline period
baselineEnd = epochParams[f'{dataPre}_baselineEnd']      # End of baseline period

# Load label data
label_evoked = np.load(allSubjs_labelEvoked_fname, allow_pickle=True)

# Read sensor data. Sensor data is stored in indivdiual subject directories, so we'll loop through subjects,
# Load the data, and append it to a dataframe.
for s in subjects:
	if s<10:
		subjectID = 'mnsbp00' + str(s)
	else:
		subjectID = 'mnsbp0' + str(s)

	# Read evoked data
	evoked = read_evoked_data(subjectID, evoked_fstem)[0]

	bad_chs.append(evoked.info['bads'])  

	# Add evoked data to list
	sens_evokeds.append(evoked)


################################################
## Get sensor snr

# Equalize channels in in the sensor data
sens_evokeds_eq = mne.equalize_channels(sens_evokeds)

# Get list of channels
channels = sens_evokeds_eq[0].info['ch_names']

# Extract evoked data to numpy array 
sens_evoked_data = np.array([evoked.data for evoked in sens_evokeds_eq])

# Compute SNR. sens_snr has shape (subjects, channels)
sens_snrs = compute_snr(sens_evoked_data, times=times, signal_window=N20m_window, noise_window=[baselineStart, baselineEnd])

# Get max snrs and best channels
sens_max_snrs, sens_max_snr_chs = get_best_channel(sens_snrs, channels)

# Check that best channels does not contain any bads
for i, ch in enumerate(sens_max_snr_chs):
	if ch in bad_chs[i]:
		raise ValueError(f'{ch} is the best ch for subject {i-1}, but was marked as a bad channel!')

# Get the most frequent best channel (across subjects)
sens_best_ch = pd.Series(sens_max_snr_chs).mode()[0]

# Get the proportion of subjects for whom best_ch was the best channel
sens_prop_best = (int(pd.Series(sens_max_snr_chs).value_counts().get(sens_best_ch, 0)) / len(sens_max_snr_chs))*100


################################################
## Get label snr

# Compute SNR for labels
label_snr = compute_snr(label_evoked, times=times, signal_window=N20m_window, noise_window=[-0.1, -0.05])

# Get mean snr for each label
label_snr_mean = np.mean(label_snr, axis=0)

# Get max snrs and best labels
label_max_snrs, label_max_snr_labels = get_best_channel(label_snr, labels)

# Get the most frequent best label (across subjects)
best_label = pd.Series(label_max_snr_labels).mode()[0]

label_prop_best = (int(pd.Series(label_max_snr_labels).value_counts().get(best_label, 0)) / len(label_max_snr_labels))*100

################################################
## Summarize results

# Sensor results
print('Sensor-level results:')
print(f'Best channel: {sens_best_ch} (Best for {sens_prop_best}% subjects)')
print(f'Range of max snrs = ({np.min(sens_max_snrs).round(2)}, {np.max(sens_max_snrs).round(2)}), Mean, Median max SNR = {np.median(sens_max_snrs).round(2)}, {np.mean(sens_max_snrs).round(2)}')

print('\n')

# Label results
print('Label-level results:')
print(f'Best label: {best_label} (Best for {label_prop_best}% subjects)')
print(f'Range of max snrs = ({np.min(label_max_snrs).round(2)}, {np.max(label_max_snrs).round(2)}), Mean, Median max SNR = {np.median(label_max_snrs).round(2)}, {np.mean(sens_max_snrs).round(2)}')

print('SNR for each label:' + ', '.join([f'{label}: {snr:.2f}' for label, snr in zip(labels, label_snr_mean)]))


