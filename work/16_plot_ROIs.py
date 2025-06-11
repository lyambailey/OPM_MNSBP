import mne
import os
import matplotlib.pyplot as plt


projectDir = '../'
subjectsDir = os.path.join(projectDir, 'subjects')
supportDir = os.path.join(projectDir, 'support_files')

MRsubject='fsaverage_with_annots'

# Read labels from fsaverage folder
labels_fs = mne.read_labels_from_annot(subject=MRsubject, parc='aparc.a2009s', subjects_dir=subjectsDir)

# Read combined label from supportDir
combined_label = mne.read_label(os.path.join(supportDir, 'combined_label_fsaverage_with_annots-lh.label'))

# Define ROIs (labels) that we are interested in

ROIs = [
    u'S_precentral-sup-part-lh',
    u'G_precentral-lh',
    u'S_central-lh',
    u'G_postcentral-lh',
    u'S_postcentral-lh',
    u'G_parietal_sup-lh',
]


ROI_labels = []
for ROI in ROIs:
	b = [x for x in labels_fs if x.name == ROI]
	if len(b) > 0:
		ROI_labels.append(b[0])

# Define colours
colors =  ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#9467BD"] # colorblind-sensitive palette 

# Visualize the label on the brain surface
brain = mne.viz.Brain(subject=MRsubject, hemi='lh', subjects_dir=subjectsDir, background='white', surf='pial', cortex='classic')  # surf='pial_semi_inflated' is an interesting option


# Overlay ROIs
for i in range(len(ROI_labels)):
	label = ROI_labels[i]
	brain.add_label(label, color=colors[i], borders=False, alpha=1)

brain.add_label(combined_label, color='black', borders=True, alpha=1)

brain.show_view('dorsal') # show top view
