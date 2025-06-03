'''Outline:
The purpose of this script is to read in saved lists of bad epochs, remove them from each subject's data, and then re-compute baseline-corrected evoked responses and TFRs. 
We will also extract mu and beta timecourses from the TFRs. This script saves group output (i.e., concatenated data for all subjects) for a single sensor. The reason
for this is that different subjects had different numbers of channels and in different orders, so it would be difficult to keep track of sensors after this stage. Data for all
sensors is saved in individual subjects' directories.

'''

# Import packages
import mne
import numpy as np
import os
import json
import ast

mne.set_log_level('ERROR') # prevents terminal output from MNE

################################################
# User input

# Task prefix
dataPre = 'MNS_shortISI'

# Subject list
subjects = [1,2,3,4,5,6,7,8,9,10,11,12]

# Write data?
writeOK = True

# Make plots?
plotOK = True

# Define sensor of interest for TFRs
pick = 'C3'

################################################
# Other parameters

n_subjects = len(subjects)

# Define parameters for the tfrs
freqs = np.arange(1, 40, 1)  
n_cycles = freqs / 3.0
mu_freqs = np.argwhere((8 < freqs) & (freqs < 15))
beta_freqs = beta_freqs = np.argwhere((15 < freqs) & (freqs < 30))

# Keep track of N epochs before rejection, and number of rejected
n_orig = []
n_rej = []

# Define empty lists to store evoked, tfrs, and mu/beta timecourses for all subjects
allSubjs_evoked = []
allSubjs_mu = []
allSubjs_beta = []
allSubjs_tfrs = []

################################################
# File paths

# Main directories
projectDir = '../'
dataDir = os.path.join(projectDir, 'proc_data')
supportDir = os.path.join(projectDir, 'support_files')
resultsDir = os.path.join(projectDir, 'results')

# Input files
epochs_fstem = '-trans-cleaned-epochs.fif' # preprocessed epochs
epochParams_fname = os.path.join(supportDir, 'all_tasks_epoch_durations.txt')  # contains baseline times
bad_subjects_fname = os.path.join(supportDir, f'{dataPre}_bad_subjects.txt') # list of bad subjects for this task


# Output files (individual subjects)
epochs_rej_fstem = '-trans-cleaned-manRej-epochs.fif'   	# epochs after manual rejection
epochsInd_rej_fstem = '-trans-cleaned-manRej-epochsInd.fif'	# epochs after manual rejection & evoked subtraction
evoked_rej_fstem = '-trans-cleaned-manRej-baseCorrected-evoked.fif'   	# evoked after manual rejection
tfrs_rej_baseCorr_fstem = '-trans-cleaned-manRej-baseCorrected-tfrs.fif'  # baseline-corrected tfrs 

# Output files (all subjects)
times_evoked_fname = os.path.join(resultsDir, f'{dataPre}-sens-evoked-times')
info_evoked_fname = os.path.join(resultsDir, f'{dataPre}-sens-evoked-info.fif')
times_tfr_fname = os.path.join(resultsDir, f'{dataPre}-sens-tfr-times')
info_tfr_fname = os.path.join(resultsDir, f'{dataPre}-sens-tfr-info.fif')

# Important: all the group output files will contain data for ONE channel only!
allSubjs_evokeds_fname = os.path.join(resultsDir, 
								f'{dataPre}-trans-cleaned-manRej-baseCorrected-evoked-{pick}_N=') # evoked after manual rejection and baseline correction, in one sens
allSubjs_tfrs_fname = os.path.join(resultsDir, 
								f'{dataPre}-trans-cleaned-manRej-baseCorrected-tfrs-{pick}_N=') # tfrs after manual rejection and baseline correction, in one sens
allSubjs_mu_fname = os.path.join(resultsDir, 
								f'{dataPre}-trans-cleaned-manRej-baseCorrected-muTCs-{pick}_N=') # mu timecourses after manual rejection and baseline correction, in one sens
allSubjs_beta_fname = os.path.join(resultsDir,
								f'{dataPre}-trans-cleaned-manRej-baseCorrected-betaTCs-{pick}_N=') # beta timecourses after manual rejection and baseline correction, in one sens

################################################
# Functions

# Define function to read txt file containing bad epochs
def read_bad_epochs(filename):

	# Create a dict to store bad epochs
	badEpochs = {}

	# Read the txt file and populate the dictionary
	with open(filename, "r") as file:
		for line in file:
			line = line.strip()
			if not line or line.startswith("#"):  # Skip comments and empty lines
				continue

			# Split the line at the '=' and remove any trailing comments
			key, value = line.split("=", 1)  # Only split on the first '='
			key = key.strip()

			# Remove any comments from the value part (after '#')
			value = value.split("#")[0].strip()

			# Append key and value to the dict
			badEpochs[key] = json.loads(value)  # json.loads parses value (a string) into a list

	return badEpochs

# Define function to read txt file containing epoch timing parameters
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

# Define function to apply baseline correction to TFR arrays
def baseline_correct_sensor_tfr(tfr_data, times, baselineStart, baselineEnd):

	# tfr_data should have shape channels x freqs x time. We'll retain the channels dimension 
	# even if we only use one channel, to avoid issues with broadcasting

	# Establish baseline indices
	baseline_idxs = np.where((times >= baselineStart) & (times <= baselineEnd))[0]

	# Extract mean baseline values (over time) for each label
	baseline_data = tfr_data[:,:,baseline_idxs]

	# Perform baseline correction: log(data / mean baseline data)
	tfr_data_corrected = np.log(tfr_data / np.mean(baseline_data, axis=2, keepdims=True))

	return tfr_data_corrected

################################################
# Do the work...

# Make sure we didn't include bad subjects
with open(bad_subjects_fname, "r") as f:
    bads = ast.literal_eval(next(line for line in f if line.strip() and not line.strip().startswith("#")))
    bads_idx = [int(bad) - 1 for bad in bads]  # Convert to zero-based index

assert set(subjects).isdisjoint(bads), \
	f'The subject list includes subjects listed in {dataPre}_bad_subjects.txt. Please check the subject list.'

# Get epoch timing parameters (we need these to baseline-correct the evoked)
epochParams = read_timing_params(epochParams_fname)
baselineStart = epochParams[f"{dataPre}_baselineStart"]
baselineEnd = epochParams[f"{dataPre}_baselineEnd"]

# Read bad epochs for all subjects into a dict. Note that is a single dict containing bad epochs for all subjects
badEpochs = read_bad_epochs(os.path.join(supportDir, f'{dataPre}_bad_epochs.txt')) 

# Loop through subjects, loading the cleaned epoch data, rejecting epochs, and saving the output
for subject in subjects:

	if subject < 10:
		subjectID = f"mnsbp00{subject}"
	else:
		subjectID = f"mnsbp0{subject}"

	print(subjectID)

	# Define data dir for this subject
	subDataDir = os.path.join(dataDir, subjectID, 'meg')

	# Files to read
	epochs_fname = os.path.join(subDataDir, f'{dataPre}{epochs_fstem}')

	# Files to write
	epochs_rej_fname = os.path.join(subDataDir, f'{dataPre}{epochs_rej_fstem}')
	epochsInd_rej_fname = os.path.join(subDataDir, f'{dataPre}{epochsInd_rej_fstem}')
	evoked_rej_fname = os.path.join(subDataDir, f'{dataPre}{evoked_rej_fstem}')
	tfrs_rej_baseCorr_fname = os.path.join(subDataDir, f'{dataPre}{tfrs_rej_baseCorr_fstem}')

	# Load the epochs
	epochs = mne.read_epochs(epochs_fname, preload=True)
	
	# Tally N epochs before rejection
	n_orig.append(len(epochs))
	
	# Get times
	times = epochs.times

	# Get bad epochs for this subject
	bad_epochs = badEpochs[subjectID]

	# Tally N bad epochs
	n_rej.append(len(bad_epochs))

	# If our selected sensor is in this subjects' list of bad channels, skip. Note that we 
	# have still included their original / rejected epoch counts in our tallies, because they 
	# are only being discarded from the sensor-level (not the source-level) anlyses
	# if pick in epochs.info['bads']:
	# 	print(f'Skipping {subjectID} because {pick} is in their list of bad channels')
	# 	n_subjects -= 1
	# 	continue

	# Drop bad epochs
	epochs_manRej = epochs.copy().drop(bad_epochs)

	# Subtract evoked response (for TFR computation)
	epochs_manRej_induced = epochs_manRej.copy().subtract_evoked()


	# For the shortISI data, we'll compute a new evoked. For MNS_longISI and buttonPress, 
	# we'll compute a TFR at one sensor
	if dataPre == "MNS_shortISI":

		# Compute evoked response for the cleaned epochs
		evoked = epochs_manRej.average()

		# Get info
		info = evoked.info

		# Apply baseline correction
		evoked.apply_baseline((baselineStart, baselineEnd))

		# Append data for our selected channel to the all-subjects list

		# If our selected sensor is in this subjects' list of bad channels, do not append. Note that we 
		# have still included their original / rejected epoch counts in our tallies, because they 
		# are only being discarded from the sensor-level (not the source-level) anlyses
		if pick in epochs.info['bads']:
			print(f'{subjectID} data will not be appended because {pick} is in their list of bad channels')
			n_subjects -= 1
			continue
		else:
			allSubjs_evoked.append(evoked.copy().pick(pick).get_data())

	elif dataPre in ['MNS_longISI', 'buttonPress', 'rest']:

		# Get info 
		info = epochs_manRej.info

		# Get the index of the pick sensor
		pick_idx = info['ch_names'].index(pick)

		# Compute tfrs from the cleaned and evoked-subtracted epochs
		tfr_data = mne.time_frequency.tfr_array_morlet(data=epochs_manRej.get_data(),
														sfreq=epochs.info['sfreq'], 
														freqs=freqs, 
														zero_mean=False,
														n_cycles=n_cycles, 
														output='power', 
														n_jobs=1)
		
		# Drop the first and last 200 ms to remove edge artifacts from the Morlet
		tfr_data = tfr_data[:, :, :, 200:-200]
		times = times[200:-200]  # update times
		
		# Average over epochs
		tfr_data_av = np.mean(tfr_data, axis=0)

		# Apply baseline correction
		tfr_data_av_corrected = baseline_correct_sensor_tfr(tfr_data_av, times, baselineStart, baselineEnd)

		# Get the data for our selected sensor
		tfr_data_av_corrected_pick = tfr_data_av_corrected[np.newaxis, pick_idx, :, :]  # using np.newaxis preserves the channel dimension

		# Extract mu and beta timecourses (i.e., the mean over each frequency range)
		mu_corrected = np.mean(tfr_data_av_corrected_pick[:,mu_freqs,:], axis=1).squeeze(1)  # remove freq dimension but preseves channel
		beta_corrected = np.mean(tfr_data_av_corrected_pick[:,beta_freqs,:], axis=1).squeeze(1)  # remove freq dimension but preseves channel

		# Append data to the all-subjects lists
		
		# If our selected sensor is in this subjects' list of bad channels, do not append. Note that we 
		# have still included their original / rejected epoch counts in our tallies, because they 
		# are only being discarded from the sensor-level (not the source-level) anlyses
		if pick in epochs.info['bads']:
			print(f'{subjectID} data will not be appended because {pick} is in their list of bad channels')
			n_subjects -= 1
			continue
		else:
			allSubjs_tfrs.append(tfr_data_av_corrected_pick)
			allSubjs_mu.append(mu_corrected)
			allSubjs_beta.append(beta_corrected)

		# Plug the baseline corrected tfr data into an epochsTFRArray objects (to be saved to disk)
		tfr_obj_av_corrected = mne.time_frequency.AverageTFR(epochs.info, tfr_data_av_corrected, times, freqs)

	# Write subject data to disk
	if writeOK:

		# Epochs (following rejections)
		epochs_manRej.save(epochs_rej_fname, overwrite=True)
		epochs_manRej_induced.save(epochsInd_rej_fname, overwrite=True)

		# Baseline-corrected mean TFRs
		if dataPre in ['MNS_longISI', 'buttonPress', 'rest']:
			tfr_obj_av_corrected.save(tfrs_rej_baseCorr_fname, overwrite=True)

		# Evoked
		if dataPre == "MNS_shortISI":
			evoked.save(evoked_rej_fname, overwrite=True)

# Convert the allSubjs_ lists to numpy arrays, so we can save them to disk.
allSubjs_evoked = np.asarray(allSubjs_evoked) 	# shape: (subjects x channels x time)
allSubjs_tfrs = np.asarray(allSubjs_tfrs) 		# shape: (subjects x channels x freqs x times)
allSubjs_mu = np.asarray(allSubjs_mu) 			# shape: (subjects x channels x times)
allSubjs_beta = np.asarray(allSubjs_beta) 		# shape: (subjects x channels x times)

# Write to disk
if writeOK:
	
	if dataPre == "MNS_shortISI":
		np.save(f'{allSubjs_evokeds_fname}{n_subjects}', allSubjs_evoked)
		np.save(times_evoked_fname, times)  # last instance of times
		info.save(info_evoked_fname) 		# last instance of info
	
	if dataPre in ['MNS_longISI', 'buttonPress', 'rest']:	
		np.save(f'{allSubjs_tfrs_fname}{n_subjects}', allSubjs_tfrs)
		np.save(f'{allSubjs_mu_fname}{n_subjects}', allSubjs_mu)
		np.save(f'{allSubjs_beta_fname}{n_subjects}', allSubjs_beta)
		np.save(times_tfr_fname, times) # last instance of times
		info.save(info_tfr_fname) 		# last instance of info

# Print total % of epochs rejected
print(f'{np.round(sum(n_rej)/sum(n_orig)*100, 2)}% epochs rejected from {dataPre} data')