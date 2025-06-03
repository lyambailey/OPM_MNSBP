# Outline: The purpose of this script is to compute Bayes factors for mu and beta ERS and ERD, in a selected sensor (ususally C3) and in a set of labels (i.e., ROIs)

# Import packages
library(BayesFactor)
library(reticulate)
library(ggplot2)
library(scales)
library(effsize)
library(abind)
np <- import('numpy') # This will trigger a prompt to set up a new environment. Enter "no"

dataPre = 'MNS_longISI' 
method_sufx = 'MNE-constrained'
n_subjects='12'
pick='C3'

# Main directories
data_path = '/media/NAS/lbailey/OPM_mnsbp/Data/results/' 
support_path = '/media/NAS/lbailey/OPM_mnsbp/Data/support_files/'

# Input files
times_fname = paste(data_path, dataPre, '-sens-tfr-times.npy', sep='') 
data_sensMu_fname = paste(data_path, dataPre, '-trans-cleaned-manRej-baseCorrected-muTCs-', 
                                pick, '_N=', n_subjects, '.npy', sep='')
data_sensBeta_fname = paste(data_path, dataPre, '-trans-cleaned-manRej-baseCorrected-betaTCs-', 
                                pick, '_N=', n_subjects, '.npy', sep='')
data_labelMu_fname = paste(data_path, dataPre, '-trans-cleaned-manRej-', 
                                method_sufx, '-baseCorrected-labelMuTCs_N=', n_subjects, '.npy', sep='')
data_labelBeta_fname = paste(data_path, dataPre, '-trans-cleaned-manRej-', 
                                method_sufx, '-baseCorrected-labelBetaTCs_N=', n_subjects, '.npy', sep='')

# Output files
bfs_fname = paste(data_path, dataPre, '_', pick, '_and_label_mu_beta_stats_N=', n_subjects, '.csv', sep='')
magns_fname = paste(data_path, dataPre, '_', pick, '_and_label_mu_beta_magnitudes_N=', n_subjects, '.csv', sep='')

# Define frequency bands
bands = c('mu', 'beta')

# Define labels
labels = c('C3', 'Superior Precentral Sulcus', 'Precentral Gyrus',
            'Central Sulcus','Postcentral Gyrus',
            'Postcentral Sulcus', 'Superior Parietal Lobule'
            )

# Define periods for baseline, ERS, and ERD
if (dataPre == 'MNS_longISI') {
  times_baseline = c(-1.0, -0.5)
  times_ERD = c(0.2, 0.4)  # delayed to avoid the MNS artifact
  times_ERS = c(0.5, 1.0)

  } else if (dataPre == 'buttonPress') {
    times_baseline = c(-2, -1.0)
    times_ERD = c(-0.25, 0.25) 
    times_ERS = c(0.5, 1.0)
} # if statement


###################
# Load data

# Load the times array and convert to r array
times = np$load(times_fname)
times = py_to_r(times)

# Define indices for baseline, ERS, and ERD
idxs_baseline = which(times >= times_baseline[1] & times <= times_baseline[2])
idxs_ERD = which(times >= times_ERD[1] & times <= times_ERD[2])
idxs_ERS = which(times >= times_ERS[1] & times <= times_ERS[2])

# Load the mu and beta data and convert each array
data_sensMu_orig = np$load(data_sensMu_fname)
data_sensMu = py_to_r(data_sensMu_orig)

data_sensBeta_orig = np$load(data_sensBeta_fname)
data_sensBeta = py_to_r(data_sensBeta_orig)

data_labelMu_orig = np$load(data_labelMu_fname)
data_labelMu = py_to_r(data_labelMu_orig)

data_labelBeta_orig = np$load(data_labelBeta_fname)
data_labelBeta = py_to_r(data_labelBeta_orig)

# Combine the sensor and label data along the 2nd dimension (sensor / label)
data_mu_sens_and_labels = abind(data_sensMu, data_labelMu, along=2)
data_beta_sens_and_labels = abind(data_sensBeta, data_labelBeta, along=2)

# Create empty dataframes to store BFs and effect magnitudes for all labels
all_bfs = data.frame()
all_magns = data.frame()

# Loop through labels. This is inefficient because we're loading the data on each loop, but it gets the job done...
for (i in seq_along(labels)) {

  label = labels[i]

  # Define a dataframe to store bfs for this label
  bfs = data.frame(label = label,
                        mu_ERD = NA, 
                        mu_ERS = NA,  # we'll compute this for programming simplicity, but we won't use it
                        beta_ERD = NA, 
                        beta_ERS = NA
  )

    magns = data.frame(label = label,
                        mu_ERD = NA, 
                        mu_ERS = NA,  # we'll compute this for programming simplicity, but we won't use it
                        beta_ERD = NA, 
                        beta_ERS = NA
    )

  # Loop through bands
  for(band in bands) {

    # Select the data for this band and label
    if (band == 'mu') {
      data = data_mu_sens_and_labels[, i, ]
    } else if (band == 'beta') {
      data = data_beta_sens_and_labels[, i, ]
    }

    # Average over each time window, preserving subjects
    data_baseline = apply(data[, idxs_baseline], 1, mean)
    data_ERD = apply(data[, idxs_ERD], 1, mean)
    data_ERS = apply(data[, idxs_ERS], 1, mean)

    # Compute magntiude as absolute value of the peak
    magn_ERD = abs(data_ERD)
    magn_ERS = abs(data_ERS)

    # Perform Bayes paired t-tests
    BF10_ERD = ttestBF(data_ERD, data_baseline, paired=TRUE, 
                        nullInterval = c(-Inf, 0))  # nullInterval = c(-Inf, 0) is a left-tailed test
    BF10_ERS = ttestBF(data_ERS, data_baseline, paired=TRUE, 
                        nullInterval = c(0, Inf))  # nullInterval = c(0, Inf) is a right-tailed test

    # Assign BF10 to bfs dataframe
    bfs[[paste0(band, '_ERD')]] = as.vector(BF10_ERD)[[1]]
    bfs[[paste0(band, '_ERS')]] = as.vector(BF10_ERS)[[1]]

    # Store magnitudes in each cell (label / effect) as a list
    magns[[paste0(band, '_ERD')]] = paste(magn_ERD, collapse=',')
    magns[[paste0(band, '_ERS')]] = paste(magn_ERS, collapse=',')

  } # bands loop

    # Append bfs to the all labels dataframe
    all_bfs = rbind(all_bfs, bfs)
    all_magns = rbind(all_magns, magns)

} # labels loop 

# Remove the mu ERS column from each dataframe
all_bfs$mu_ERS = NULL
all_magns$mu_ERS = NULL

# Write bfs to disk
write.csv(all_bfs, file=bfs_fname, row.names=FALSE)

# Write magnitudes to disk. Each cell will be enclosed in quotes, to preserve the lists without disrupting csv formatting
write.csv(all_magns, file=magns_fname, row.names=FALSE, quote=TRUE)

