'''
Outline:
The purpose of this script is to perform source localization on sensor-level evoked data and return an evoked response
for each subject, in each of six pre-defined ROIs. This script is only intended to be run on the MNS_shortISI data. 

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
dataPre = 'MNS_shortISI'

# Subject list
subjects = [1]#,2,3,4,5,6,7,8,9,10,11,12]  # check this does not includes bads

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
epochs_fstem = f'{dataPre}-trans-cleaned-manRej-epochs.fif'
evoked_fstem = f'{dataPre}-trans-cleaned-manRej-baseCorrected-evoked.fif'
combined_label_fname = os.path.join(supportDir, 'combined_label_fsaverage_with_annots-lh.label')
bad_subjects_fname = os.path.join(supportDir, f'{dataPre}_bad_subjects.txt')

# Suffix for source localization output
if constrained:
	method_sufx = f'{method}-constrained'
else:
	method_sufx = f'{method}-unconstrained'

# Output files (individual subjects)
stcEvoked_fstem = f'{dataPre}-trans-cleaned-manRej-baseCorrected-{method_sufx}-stcEvoked'

# Output files (all subjects)
n_subjects = len(subjects)
allSubjs_stcEvoked_fname = os.path.join(resultsDir, 
									f'{dataPre}-trans-cleaned-manRej-baseCorrected-{method_sufx}-stcEvoked_N={n_subjects}')
allSubjs_labelEvoked_fname = os.path.join(resultsDir, 
									f'{dataPre}-trans-cleaned-manRej-baseCorrected-{method_sufx}-labelEvoked_N={n_subjects}')


################################################
# Do the work...

# Define lists to store data for all subjects
allSubjs_stcs = []
allSubjs_labelData = []

# Make sure we didn't include bad subjects
with open(bad_subjects_fname, "r") as f:
    bads = ast.literal_eval(next(line for line in f if line.strip() and not line.strip().startswith("#")))
    bads_idx = [int(bad) - 1 for bad in bads]  # Convert to zero-based index

assert set(subjects).isdisjoint(bads_idx), \
	f'The subject list includes subjects listed in {dataPre}_bad_subjects.txt. Please check the subject list.'

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

# Loop though subjects, performing source localization on evoked data and extracting label time courses. 
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
	evoked_fname = os.path.join(subOutDir, evoked_fstem)
	MRItrans_fname = os.path.join(projectDir, 'proc_data', subjectID, 'digi', 'head_mri-trans.fif')
	bemSol_fname = os.path.join(subjectsDir, MRsubject, 'bem', MRsubject + '-5120-5120-5120-bem-sol.fif')

	# Output files
	stcEvoked_fname = os.path.join(subOutDir, stcEvoked_fstem)
	
	# Read files
	evoked = mne.read_evokeds(evoked_fname)[0]		# evoked
	epochs = mne.read_epochs(epochs_fname)			# epochs
	MRItrans = mne.read_trans(MRItrans_fname)		# head->MRI transform
	bemSol = mne.read_bem_solution(bemSol_fname) 	# bem solution

	# Define label for constraining localization
	if constrained:

		# Morph anatomical labels to this subject
		combined_label_morph = combined_label.copy().morph(subject_from='fsaverage_with_annots', 
									   		subject_to=MRsubject, subjects_dir=subjectsDir)
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
	epochs_cov = mne.compute_covariance(epochs) 

	# Create source space
	src = mne.setup_source_space(MRsubject, subjects_dir=subjectsDir, n_jobs=-1)  	# n_jobs=-1 uses all cores for parallel processing. 
																					# This speeds things up substantially

	# Create forward solution
	fwd = mne.make_forward_solution(evoked_fname, trans=MRItrans, src=src, bem=bemSol, meg=True)

	# Create inverse operator
	inv = mne.minimum_norm.make_inverse_operator(evoked.info, forward=fwd, noise_cov=epochs_cov)

	# Compute stc
	stc_evoked = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=lambda2, method=method, 
											pick_ori="normal", label=label)

	# For compatibility with the other scripts (i.e., the TFR anlayses), wrap the stc in a numpy array (we'll 
	# save this array to disk later). The stc can be easily unpacked later using stc = stc_array.tolist()[0]
	stc_arr = np.asarray(stc_evoked)

	# Extract label evoked
	labels_evoked = stc_evoked.extract_label_time_course(labels=ROI_labels, src=src, mode='pca_flip') # shape: labels x time

	# Append data to lists
	allSubjs_stcs.append(stc_evoked)
	allSubjs_labelData.append(labels_evoked)

	# Write the data to disk
	if writeOK:
		np.save(stcEvoked_fname, stc_arr)

	print('Done!')
	print('######################################################')

# Convert all subjects data to numpy arrays and write to disk
allSubjs_stcs = np.asarray(allSubjs_stcs)
allSubjs_labelData = np.asarray(allSubjs_labelData)

print(f'All subjects stcs shape: {allSubjs_stcs.shape}')
print(f'All subjects labelData shape: {allSubjs_labelData.shape}')

if writeOK:
	np.save(allSubjs_stcEvoked_fname, allSubjs_stcs)
	np.save(allSubjs_labelEvoked_fname, allSubjs_labelData)
