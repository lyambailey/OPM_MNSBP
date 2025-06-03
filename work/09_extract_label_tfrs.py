'''
Outline:
The purpose of this script is to perform source localization on sensor-level epoch data and return an array of tfrs (one per epoch)
for each subject, in each of six pre-defined ROIs. This script is only intended to be run on the MNS_longISI or buttonPress data. 

'''

# Import packages
import mne
import numpy as np
import os
import ast

mne.set_log_level()
# mne.set_log_level('ERROR') # prevents terminal output

################################################
# User input

# Task prefix
dataPre = 'MNS_longISI'

# Subject list
subjects = [1,2,3,4,5,6,7,8,9,10,11,12]

# Write data?
writeOK = True

################################################
# Source localization parametrs
method = 'MNE'
constrained = True # Constrain source estimation to our combined label?
snr = 3.0 
lambda2 = 1.0 / snr**2

# Define ROIs (labels) that we are interested in
ROIs = [u'S_precentral-sup-part-lh', u'G_precentral-lh',
		u'S_central-lh', u'G_postcentral-lh',
		u'S_postcentral-lh', u'G_parietal_sup-lh']

# Label extraction parameters
label_extract_mode = 'pca_flip'

# TFR parameters
freqs = np.arange(1, 40, 1)  
n_cycles = freqs / 3.0
mu_freqs = np.argwhere((8 < freqs) & (freqs < 15))
beta_freqs = beta_freqs = np.argwhere((15 < freqs) & (freqs < 30))

################################################
# Define file paths 

# Main directories
projectDir = '../'
dataDir = os.path.join(projectDir, 'raw_data')
outputDir = os.path.join(projectDir, 'proc_data')
resultsDir = os.path.join(projectDir, 'results')
supportDir = os.path.join(projectDir, 'support_files')
subjectsDir = os.path.join(projectDir, 'subjects')

# Input files
epochs_fstem = f'{dataPre}-trans-cleaned-manRej-epochsInd.fif'
combined_label_fname = os.path.join(supportDir, 'combined_label_fsaverage_with_annots-lh.label')
epochParams_fname = os.path.join(supportDir, 'all_tasks_epoch_durations.txt')  # contains baseline times
bad_subjects_fname = os.path.join(supportDir, f'{dataPre}_bad_subjects.txt')
times_fname = os.path.join(resultsDir, f'{dataPre}-sens-tfr-times.npy')

# Prefix for source localization output
if constrained:
	method_sufx = f'{method}-constrained'
else:
	method_sufx = f'{method}-unconstrained'

# Output files (individual subjects)
epochs_stc_fstem = f'{dataPre}-trans-cleaned-manRej-{method_sufx}-stcEpochs'

# Output files (all subjects)
n_subjects = len(subjects)
allSubjs_stcEpochs_fname = os.path.join(resultsDir, 
										 f'{dataPre}-trans-cleaned-manRej-{method_sufx}-stcEpochs_N={n_subjects}')  # epoch stcs
allSubjs_labelTFRcorr_fname = os.path.join(resultsDir, 
										f'{dataPre}-trans-cleaned-manRej-{method_sufx}-baseCorrected-labelTFRs_N={n_subjects}') # baseline-corrected label tfrs
allSubjs_labelTFRuncorr_fname = os.path.join(resultsDir, 
										f'{dataPre}-trans-cleaned-manRej-{method_sufx}-uncorrected-labelTFRs_N={n_subjects}') # uncorrected label tfrs
allSubjs_labelMu_fname = os.path.join(resultsDir,
										f'{dataPre}-trans-cleaned-manRej-{method_sufx}-baseCorrected-labelMuTCs_N={n_subjects}') # label mu timecourses
allSubjs_labelBeta_fname = os.path.join(resultsDir,
										f'{dataPre}-trans-cleaned-manRej-{method_sufx}-baseCorrected-labelBetaTCs_N={n_subjects}') # label beta timecourses
ROIs_fname = os.path.join(resultsDir,
							f'{dataPre}_evoked_ROIs.npy')

################################################
# Functions

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
def baseline_correct_label_tfr(tfr_data, times, baselineStart, baselineEnd):

	# tfr_data should have shape labels x freqs x time. 

	# Establish baseline indices
	baseline_idxs = np.where((times >= baselineStart) & (times <= baselineEnd))[0]

	# Extract mean baseline values (over time) for each label
	baseline_data = tfr_data[:,:,baseline_idxs]

	# Perform baseline correction: log(data / mean baseline data)
	tfr_data_corrected = np.log(tfr_data / np.mean(baseline_data, axis=2, keepdims=True))

	return tfr_data_corrected

################################################
# Do the work...

print(f'Processing {n_subjects} subjects...')

# Define lists to store data for all subjects
allSubjs_stcs = []
allSubjs_labelTFRsUncorr = []
allSubjs_labelTFRsCorr = []
allSubjs_labelMu = []
allSubjs_labelBeta = []

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

# Get times
times = np.load(times_fname)

# Read anatomical labels from the fsaverage annotation file
labels_fs = mne.read_labels_from_annot(subject='fsaverage_with_annots', 
                                        parc='aparc.a2009s', subjects_dir=subjectsDir)

# Load the combined label/ROI for constrained source estimation (we'll only use this if constrained==True)
if constrained:
	
	if os.path.exists(combined_label_fname):
		print('Reading combined label from disk...')
		combined_label = mne.read_label(combined_label_fname, subject='fsaverage_with_annots')
		combined_label.name = 'combined_label'

	else:
		print('Creating combined label for constrained source estimation...')

		# Make a list of labels corresponding to our ROIs.
		ROI_labels_for_comb = []
		for ROI in ROIs:
			b = [x for x in labels_fs if x.name == ROI]
			if len(b) > 0:
				ROI_labels_for_comb.append(b[0])

		# Combine the labels into a single label
		combined_label = mne.Label(hemi='lh', subject='fsaverage_with_annots')

		verts = []
		for label in ROI_labels_for_comb:
			verts.append(len((label.vertices)))
			combined_label = combined_label.__add__(label)

		# Re-set the name
		combined_label.name = 'combined_label'

		# Verfiy that the number of vertices in combined_label is 
		# equal to the sum of the vertices in the individual labels
		assert sum(verts) == len(combined_label.vertices), \
			"Number of vertices in combined label does not match the sum of vertices in individual labels."

		# Write the combined label to disk
		combined_label.save(combined_label_fname)

print(f'Processing {n_subjects} subjects...')

# Loop though subjects, performing source localization on epoch data and extracting label time courses. 
for s in subjects:
	if s<10:
		subjectID = 'mnsbp00' + str(s)
	else:
		subjectID = 'mnsbp0' + str(s)
	
	print(subjectID)

	# ID for the scaled-to-subject MRI folder
	MRsubject = f'fsaverage_scaled_to_{subjectID}'

	# Subject directories
	subOutDir = os.path.join(outputDir, subjectID, 'meg')
	subMRpath = os.path.join(subjectsDir, MRsubject)

	# Input files
	epochs_fname = os.path.join(subOutDir, epochs_fstem)
	MRItrans_fname = os.path.join(projectDir, 'proc_data', subjectID, 'digi', 'head_mri-trans.fif')
	bemSol_fname = os.path.join(subjectsDir, MRsubject, 'bem', MRsubject + '-5120-5120-5120-bem-sol.fif')

	# Output files
	epochs_stc_fname = os.path.join(subOutDir, epochs_stc_fstem)
	
	# Read files
	epochs = mne.read_epochs(epochs_fname)			# epochs
	MRItrans = mne.read_trans(MRItrans_fname)		# head->MRI transform
	bemSol = mne.read_bem_solution(bemSol_fname) 	# bem solution

	# Morph anatomical labels to this subject
	combined_label_morph = combined_label.copy().morph(subject_from='fsaverage_with_annots', 
									   		subject_to=MRsubject, subjects_dir=subjectsDir)
	if constrained:
		label=combined_label_morph   # used when applying inverse operator
	else:
		label=None
	
	labels_morph = mne.morph_labels(labels=labels_fs, subject_to=MRsubject, 
											subject_from='fsaverage_with_annots', 
											subjects_dir=subjectsDir)
	
	# Make a list of (morphed) labels corresponding to our ROI list
	ROI_labels = []
	for ROI in ROIs:
		b = [x for x in labels_morph if x.name == ROI]
		if len(b) > 0:
			ROI_labels.append(b[0])	

	# Get noise covariance from epochs
	epochs_cov = mne.compute_covariance(epochs) # old version
	
	# Create source space
	src = mne.setup_source_space(MRsubject, subjects_dir=subjectsDir)

	# Create forward solution
	fwd = mne.make_forward_solution(epochs_fname, trans=MRItrans, src=src, bem=bemSol, meg=True)

	# Create inverse operator
	inv = mne.minimum_norm.make_inverse_operator(epochs.info, forward=fwd, noise_cov=epochs_cov)

	# Compute epoch stcs
	stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=lambda2, method=method, 
										pick_ori="normal", label=label)

	# stcs is a list of stc objects (one per epoch). Convert stcs to a numpy array so that we can 
	# save it to disk. The list can be easily unpacked later, using stcs = stc_arr.tolist()
	stc_arr = np.asarray(stcs)

	# Extract label time courses from each epoch stc
	timecourse_data = []

	for stc in stcs:  # one stc per epoch

		# Extract the timecourses from all labels in our list of ROIs
		timecourses = stc.extract_label_time_course(labels=ROI_labels, src=src, mode='pca_flip') # shape: labels x time

		timecourse_data.append(timecourses)

	# Concatenate the timecourses into an array with shape (epochs x labels x time)
	timecourse_data = np.array(timecourse_data) 

	# Compute tfrs. Note that the tfr_array_morlet assumes the input data has shape (n_epochs, n_channels, n_times). 
	# timecourse_data has shape (n_epochs, n_labels, n_times), so the output will treat labels as channels.
	label_tfrs = mne.time_frequency.tfr_array_morlet(timecourse_data, sfreq=epochs.info['sfreq'], zero_mean=False, 
												freqs=freqs, n_cycles=n_cycles, output='power')
	
	# Drop the first and last 200 ms to remove edge artifacts from the Morlet. Note that times has already been corrected by a previous script
	label_tfrs = label_tfrs[:,:,:,200:-200] 
	
	# Average over epochs
	label_tfrs_av = np.mean(label_tfrs, axis=0)	
	
	# Apply baseline correction
	label_tfrs_av_corrected = baseline_correct_label_tfr(label_tfrs_av, times, baselineStart, baselineEnd)

	# Extract mu and beta timecourses (i.e., the mean over each frequency range)
	label_mu_corrected = np.mean(label_tfrs_av_corrected[:,mu_freqs,:], axis=1).squeeze()
	label_beta_corrected = np.mean(label_tfrs_av_corrected[:,beta_freqs,:], axis=1).squeeze()


	# Append data to lists
	allSubjs_stcs.append(stc)
	allSubjs_labelTFRsUncorr.append(label_tfrs_av)
	allSubjs_labelTFRsCorr.append(label_tfrs_av_corrected)
	allSubjs_labelMu.append(label_mu_corrected)
	allSubjs_labelBeta.append(label_beta_corrected)

	# Write the data to disk
	if writeOK:
		np.save(epochs_stc_fname, stc_arr)

	print('Done!')
	print('######################################################')

# # Convert all subjects data to numpy arrays and write to disk
allSubjs_stcs = np.asarray(allSubjs_stcs)
allSubjs_labelTFRsUncorr = np.asarray(allSubjs_labelTFRsUncorr)
allSubjs_labelTFRsCorr = np.asarray(allSubjs_labelTFRsCorr)
allSubjs_labelMu = np.asarray(allSubjs_labelMu)
allSubjs_labelBeta = np.asarray(allSubjs_labelBeta)

print(f'All subjects stcs shape: {allSubjs_stcs.shape}')
print(f'All subjects label TFRs shape: {allSubjs_labelTFRsCorr.shape}')
print(f'All subjects label mu shape: {allSubjs_labelMu.shape}')
print(f'All subjects label beta shape: {allSubjs_labelBeta.shape}')

if writeOK:
	np.save(allSubjs_stcEpochs_fname, allSubjs_stcs)
	np.save(allSubjs_labelTFRuncorr_fname, allSubjs_labelTFRsUncorr)
	np.save(allSubjs_labelTFRcorr_fname, allSubjs_labelTFRsCorr)
	np.save(allSubjs_labelMu_fname, allSubjs_labelMu)
	np.save(allSubjs_labelBeta_fname, allSubjs_labelBeta)
	np.save(ROIs_fname, ROIs)
