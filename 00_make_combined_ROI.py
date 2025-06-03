'''
Outline:
The purpose of this script is to combine labels from different ROIs into a single label. We'll use this combined label down the line to contrain source localization.
'''
import mne
import os

# Define main directories 
projectDir = '../'
subjectsDir = os.path.join(projectDir, 'subjects')
supportDir = os.path.join(projectDir, 'support_files')

# Define path for writing the combined label
combined_label_fname = os.path.join(supportDir, 'combined_label_fsaverage_with_annots.label')

# Read anatomical labels from the fsaverage annotation file
labels_fs = mne.read_labels_from_annot(subject='fsaverage_with_annots', 
                                        parc='aparc.a2009s', subjects_dir=subjectsDir)

# Define ROIs (labels) that we are interested in
ROIs = [
    u'S_precentral-sup-part-lh',
    u'G_precentral-lh',
    u'S_central-lh',
    u'G_postcentral-lh',
    u'S_postcentral-lh',
    u'G_parietal_sup-lh',
]

# Make a list of labels corresponding to our ROIs.
ROI_labels = []
for ROI in ROIs:
	b = [x for x in labels_fs if x.name == ROI]
	if len(b) > 0:
		ROI_labels.append(b[0])

# Combine the labels into a single label
combined_label = mne.Label(hemi='lh', subject='fsaverage_with_annots')

verts = []
for label in ROI_labels:
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