# Outline: The purpose of this script is to compute Bayes factors for mu and beta ERS and ERD, in a selected sensor (ususally C3) and in a set of labels (i.e., ROIs)

# Import packages
library(BayesFactor)
library(reticulate)
library(ggplot2)
library(scales)
library(effsize)
library(abind)
np <- import('numpy') # This will trigger a prompt to set up a new environment. Enter "no"

dataPre = 'MNS_shortISI' 
method_sufx = 'MNE-constrained'
pick='C3'

# Main directories
data_path = '/media/NAS/lbailey/OPM_mnsbp/Data/results/' 
support_path = '/media/NAS/lbailey/OPM_mnsbp/Data/support_files/'

# Input files
times_fname = paste(data_path, dataPre, '-sens-evoked-times.npy', sep='') 
data_sens_evoked_fname = paste(data_path, dataPre, 
              '-trans-cleaned-manRej-baseCorrected-evoked-', pick, '_N=11.npy', sep='')
data_labels_evoked_fname = paste(data_path, dataPre, 
              '-trans-cleaned-manRej-baseCorrected-', method_sufx, '-labelEvoked_N=12.npy', sep='')
              
# Output files
bfs_fname = paste(data_path, dataPre, '_', pick, '_and_label_evoked_stats.csv', sep='')
magns_fname = paste(data_path, dataPre, '_', pick, '_and_label_evoked_magnitudes.csv', sep='')

# Define labels
labels = c('C3', 'Superior Precentral Sulcus', 'Precentral Gyrus',
            'Central Sulcus','Postcentral Gyrus',
            'Postcentral Sulcus', 'Superior Parietal Lobule'
            )

# Define periods for baseline and peaks: 20 ms, 25 ms, 60 ms
times_baseline = c(-0.2, -0.1)
times_win20 = c(0.015, 0.025)  # 10 ms window around 20 ms peak
times_win35 = c(0.030, 0.040) # 10 ms window around 35 ms peak
times_win60 = c(0.055, 0.065) # 10 ms window around 60 ms peak


###################
# Load data

# Load the times array and convert to r array
times = np$load(times_fname)
times = py_to_r(times)

# Define indices for the baseline window and our 3 peaks
idxs_baseline = which(times >= times_baseline[1] & times <= times_baseline[2]) 
idxs_win20 = which(times >= times_win20[1] & times <= times_win20[2])
idxs_win35 = which(times >= times_win35[1] & times <= times_win35[2])
idxs_win60 = which(times >= times_win60[1] & times <= times_win60[2])

# Load the sens and labels data and convert to r array
data_sens = np$load(data_sens_evoked_fname)
data_sens = py_to_r(data_sens)

data_labels = np$load(data_labels_evoked_fname)
data_labels = py_to_r(data_labels)

# Create empty dataframes to store BFs and effect magnitudes for all labels
all_bfs = data.frame()
all_magns = data.frame()

# Loop through labels. This is inefficient because we're loading the data on each loop, but it gets the job done...
for (i in seq_along(labels)) {

  label = labels[i]

  # Select the data for this label (or if i==1, use the sensor data)
  if (i == 1) {
    data = data_sens[, i, ]

    } else {
      data = data_labels[, i-1, ]
    }

  # Define a dataframe to store bfs for this label
  bfs = data.frame(label = label,
                      SEF20 = NA, 
                      SEF35 = NA, 
                      SEF60 = NA
    )

  magns = data.frame(label = label,
                    SEF20 = NA, 
                    SEF35 = NA, 
                    SEF60 = NA
    )

  # Find the index for the max value in each time window, averaged across subjects. 
  idx_peak20 = which.max(abs(apply(data[, idxs_win20], 2, mean))) + min(idxs_win20)
  idx_peak35 = which.max(apply(data[, idxs_win35], 2, mean)) + min(idxs_win35)
  idx_peak60 = which.max(apply(data[, idxs_win60], 2, mean)) + min(idxs_win60)

  # Create a very narrow time window (peak +/- 1 ms) around the peak
  idxs_peak20_3ms = c(idx_peak20 - 1, idx_peak20, idx_peak20 + 1)
  idxs_peak35_3ms = c(idx_peak35 - 1, idx_peak35, idx_peak35 + 1)
  idxs_peak60_3ms = c(idx_peak60 - 1, idx_peak60, idx_peak60 + 1)

  # Average over each 3ms time window, preserving subjects. Also average over the baseline window
  data_baseline = apply(data[, idxs_baseline], 1, mean)
  data_peak20 = apply(data[, idxs_peak20_3ms], 1, mean)
  data_peak35 = apply(data[, idxs_peak35_3ms], 1, mean)
  data_peak60 = apply(data[, idxs_peak60_3ms], 1, mean)

  # Compute magntiude of each response: abs(peak - baseline)
  magn_peak20 = abs(data_peak20 - data_baseline)
  magn_peak35 = abs(data_peak35 - data_baseline)
  magn_peak60 = abs(data_peak60 - data_baseline)

  # Determine the sign of the first peak, and select left/right tailed test.
  # The tail for the other two peaks will be the opposite of the first peak
  if (sign(mean(data_peak20))==1) {
    peak20_nullInterval = c(0, Inf)
    peak35_nullInterval = c(-Inf, 0)
    peak60_nullInterval = c(-Inf, 0)
  } else {
    peak20_nullInterval = c(-Inf, 0)
    peak35_nullInterval = c(0, Inf)
    peak60_nullInterval = c(0, Inf)
  }


  # Perform Bayes paired t-tests
  BF10_peak20 = ttestBF(data_peak20, data_baseline, paired=TRUE, 
              nullInterval = peak20_nullInterval) 
  BF10_peak35 = ttestBF(data_peak35, data_baseline, paired=TRUE,
              nullInterval = peak35_nullInterval)
  BF10_peak60 = ttestBF(data_peak60, data_baseline, paired=TRUE,
              nullInterval = peak60_nullInterval)

  # Assign BF10 to bfs dataframe
  bfs$SEF20 = as.vector(BF10_peak20)[[1]]
  bfs$SEF35 = as.vector(BF10_peak35)[[1]]
  bfs$SEF60 = as.vector(BF10_peak60)[[1]]

  # Store magnitudes in each cell (label / effect) as a list
  magns$SEF20 = paste(magn_peak20, collapse=',')
  magns$SEF35 = paste(magn_peak35, collapse=',')
  magns$SEF60 = paste(magn_peak60, collapse=',')

  # Append bfs to the all labels dataframe
  all_bfs = rbind(all_bfs, bfs)
  all_magns = rbind(all_magns, magns)

} # labels loop 
all_bfs


# Write bfs to disk
write.csv(all_bfs, file=bfs_fname, row.names=FALSE)

# Write magnitudes to disk. Each cell will be enclosed in quotes, to preserve the lists without disrupting csv formatting
write.csv(all_magns, file=magns_fname, row.names=FALSE, quote=TRUE)
