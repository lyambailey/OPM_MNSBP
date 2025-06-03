'''
Outline:
The purpose of this script is to perform basic preprocessing on the registered-to-standard sensor data. 
Note that this script handles data from one task (MNS_shortISI, MNS_longISI, or buttonPress) at a time!
'''

# Import packages
import mne
import numpy as np
import pandas as pd
import os
import copy
import scipy.signal as ss
import scipy.stats as sstats
from sklearn import linear_model
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import json

################################################
# User input

# Task prefix
rawPre = 'buttonPress'

# Subject list
subjects = [5]#[1,2,3,4,5,6,7,8,9,10,11,12]

# Write data?
writeOK = True

################################################
# Define file paths and preprocessing parameters

# Main directories
projectDir = '../'
dataDir = os.path.join(projectDir, 'proc_data')
resultsDir = os.path.join(projectDir, 'results')
supportDir = os.path.join(projectDir, 'support_files')
subjectsDir = os.path.join(projectDir, 'subjects')

# Input filenames
badChannelsFile = os.path.join(supportDir, f'{rawPre}_bad_channels.txt')
eventsFile = os.path.join(supportDir, f'{rawPre}_stimChannel_threshold_and_bad_events.txt') # only exists for buttonPress
epochParamsFile = os.path.join(supportDir, 'all_tasks_epoch_durations.txt')
raw_fname = f'{rawPre}-trans-raw.fif'

# Output filenames
cleaned_raw_fstem = f'{rawPre}-trans-cleaned-raw.fif'
cleaned_epochs_fstem = f'{rawPre}-trans-cleaned-epochs.fif'
cleaned_evoked_fstem = f'{rawPre}-trans-cleaned-evoked.fif'
ica_fstem = f'{rawPre}-trans-cleaned-epochs-ica.fif'
raw_ampSpec_fstem = f'{rawPre}-raw-ampSpec.npy'
raw_cleaned_ampSpec_fstem = f'{rawPre}-raw-cleaned-ampSpec.npy'
ampSpec_freq_fstem = f'{rawPre}-ampSpec-freq.npy'

# Filtering 
l_freq1=3
h_freq1=150
notch_freqs = [60, 120, 180] # line noise and harmonics

# Other parameters
project = True	# apply HFC?
shcOrder=1
windowDuration = 4  # seconds
n_fft = 2000

# Define params for making fixed-length events (only relevant to rest data)
crop_window = 20 # time to crop from the start and end of the recording, in seconds
duration = 2 # the duration between events (i.e., the ISI), in seconds
Nevents = 80 # number of events to create, equal to the number of trials in buttonPress and longISI tasks

################################################
# Read support files

# Define dicts to store parameters for all subjects
epochParams = {}
badChannels = {}
event_thresholds_and_bads = {}

# Epoch timing parameters
with open(epochParamsFile, "r") as file:
    for line in file:
        # Skip lines that start with '#' or are empty
        if line.strip() and not line.startswith("#"):
            # Split the line at '='
            key, value = line.split("=", 1)
            key = key.strip()  # Remove whitespace around the key
            value = float(value.strip())  # Convert the value to float
            epochParams[key] = value  # Store in the dictionary

# Get timing info for this task only
epochStart = epochParams[f"{rawPre}_epochStart"]
epochEnd = epochParams[f"{rawPre}_epochEnd"]
baselineStart = epochParams[f"{rawPre}_baselineStart"]
baselineEnd = epochParams[f"{rawPre}_baselineEnd"]
evokedStart = epochParams[f"{rawPre}_evokedStart"]
evokedEnd = epochParams[f"{rawPre}_evokedEnd"]

# Bad channels
with open(badChannelsFile, "r") as file:
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
		badChannels[key] = json.loads(value)  # json.loads parses value (a string) into a list

# Thresholds for event detection (only applicable to buttonPress data)
if rawPre == 'buttonPress':

	# Read file containing event threholds for each subject
	with open(eventsFile, "r") as file:
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
			event_thresholds_and_bads[key] = json.loads(value)  # json.loads parses value (a string) into a list


################################################
# Functions

def getDefaultSensorLocations(supportDir, digName):
	# Get file with default sensor locations for this sensor layout
	digDf = pd.read_excel(os.path.join(supportDir, digName))

	# Make the array to set sensor locations
	locColumns = ['sensor_x', 'sensor_y', 'sensor_z', 'ex_i',
		   'ex_j', 'ex_k', 'ey_i', 'ey_j', 'ey_k', 'ez_i', 'ez_j', 'ez_k',
		   'x cell\n(original)', 'y cell\n(original)', 'z cell\n(original)' ]
	locArray = np.asarray(digDf.loc[:,locColumns])

	sensorNames = digDf['Sensor Names'].to_list()

	return locArray, sensorNames

def updateSensorNamesAndLocations(raw, channelSubset, sd, locArray, sensorNames):

	# Determine the relevant indices for channels of interest
	channelSubsetIndex = [sensorNames.index(i) for i in channelSubset]
	numChannels = len(channelSubsetIndex)

	# Take only relevant locations
	locSubArray = locArray[channelSubsetIndex,:]

	# Update locations using measured sleeve depths on helmet
	sensX = (locSubArray[:,12]-((sd)/1000)*locSubArray[:,9])*-1
	sensY = (locSubArray[:,13]-((sd)/1000)*locSubArray[:,10])*-1
	sensZ = (locSubArray[:,14]+((sd)/1000)*locSubArray[:,11])
	locSubArray[:,0] = sensX
	locSubArray[:,1] = sensY
	locSubArray[:,2] = sensZ

	# Drop last three columns (no longer needed)
	locSubArray = locSubArray[:,0:12]

	# Replace locations in raw.info
	newRaw = raw.copy()
	offsetIndex = 0
	for c in np.arange(numChannels):
		channelIndex = c+offsetIndex
		newRaw.info['chs'][channelIndex]['loc'] = locSubArray[c,0:12]

	# Rename channels based on the dataset (to avoid duplicate sensor names in the combined evoked)
	numSensors = len(channelSubset)
	channelNames = newRaw.info['ch_names'][offsetIndex:numSensors]
	ctr = 0
	nameDict = {}
	for ch in channelNames:
		nameDict[ch] = channelSubset[ctr]
		ctr += 1
	newRaw.rename_channels(nameDict)

	return newRaw, locSubArray

def referenceArrayRegression(raw_filter, opmChannels, sensorChannels, refChannels):

	# Window data (1 second cosine) to clean out high-pass edge effects
	opmData = raw_filter.get_data()[opmChannels]

	# Remove signals related to reference signals via regression
	sensorData = opmData[sensorChannels,:]
	referenceData = opmData[refChannels,:]

	numSensors = len(sensorChannels)
	regressData = copy.copy(sensorData)
	for i in np.arange(numSensors):
		# Put data into a pandas dataframe
		data = {'sensor': sensorData[i,:],
				'Xref': referenceData[0,:],
				'Yref': referenceData[1,:],
				'Zref': referenceData[2,:],
				}
		df = pd.DataFrame(data)
		x = df[['Xref','Yref', 'Zref']]
		y = df['sensor']
		# Run multi-variable regression
		regr = linear_model.LinearRegression()
		regr.fit(x, y)
		# Extract cleaned sensor data 
		regressData[i,:] = sensorData[i,:] - regr.coef_[0]*referenceData[0,:] - regr.coef_[1]*referenceData[1,:] - regr.coef_[2]*referenceData[2,:]

	# Put cleaned data into a raw_regress object
	allData = raw_filter.get_data()
	allData[sensorChannels,:] = regressData
	raw_regressed = mne.io.RawArray(allData, raw_filter.info)

	return raw_regressed

def detrendData(raw, opmChannels):

	# Linear regression to baseline correct and detrend
	opmData = raw.copy().pick('mag').get_data()
	opmDataBaseCorr = copy.copy(opmData)
	for i in np.arange(opmData.shape[0]):
		result = sstats.linregress(raw.times,opmData[i,:])
		modelFit = result.intercept + result.slope*raw.times
		opmDataBaseCorr[i,:] = opmData[i,:] - modelFit

	# Put detrended data into a raw object
	allData = raw.get_data()
	allData[opmChannels,:] = opmDataBaseCorr
	raw_detrend = mne.io.RawArray(allData, raw.info)

	return raw_detrend

def windowData(raw, opmChannels, winDur):

	# Window data (1 second cosine) to clean out high-pass edge effects
	opmData = raw.copy().pick('mag').get_data()
	numSamples = opmData.shape[1]
	numChannels = opmData.shape[0]
	fs = raw.info['sfreq']
	cosTimes =  np.arange(winDur*fs)/fs
	cosWin = np.expand_dims(-0.5*np.cos(2*np.pi*0.5*cosTimes/winDur)+0.5,0)	# half a cosine
	middleWin = np.ones((1,numSamples-2*len(cosTimes)))
	fullWindow = np.tile(np.hstack((cosWin, middleWin, cosWin[:,::-1])), (numChannels,1))
	windowedData = opmData*fullWindow

	# Put detrended data into a raw object
	allData = raw.get_data()
	allData[opmChannels,:] = windowedData
	raw_win = mne.io.RawArray(allData, raw.info)

	return raw_win

def getEvents_MNS(raw, voltageThreshold):

	# Get the stim channel, which has square waves for each stimulus
	stimChan = mne.channel_indices_by_type(raw.info)['stim'][0]
	stimData = raw.get_data()[stimChan,:]

	# Take the difference
	stimDiff = np.diff(stimData)
		
	# Find positive peaks in the difference signal. Using stimDiff marks peak onset, and 
	# using -stimDiff marks peak offset. We'll use *onset* for MNS. 
	peakIndex = ss.find_peaks(stimDiff, height=voltageThreshold, distance=100)[0]
	numTrials = len(peakIndex)
	# print('There are ' + str(numTrials) + ' trials.')

	# Make a list of the *onset* time of each stimulus
	events = np.vstack((peakIndex, np.zeros(numTrials)))
	events = np.vstack((events, np.ones(numTrials))).T.astype(int)

	return events	

def getEvents_buttonPress(raw, voltageThreshold, badEvents):  

	# Get the stim channel, which has square waves for each stimulus
	stimData = raw.get_data()[-1,:] # stim channel is last channel in raw data

	# Take the difference
	stimDiff = np.diff(stimData)
		
	# Find positive peaks in the difference signal. Using stimDiff marks peak onset, and 
	# using -stimDiff marks peak offset. We'll use *offset* for buttonPress. 
	peakIndex = ss.find_peaks(-stimDiff, height=voltageThreshold, distance=100)[0]

	# Drop bad events
	peakIndex = np.delete(peakIndex, badEvents)
	numTrials = len(peakIndex)
	# print('There are ' + str(numTrials) + ' trials.')

	# Make a list of the *offset* time of each button press
	events = np.vstack((peakIndex, np.zeros(numTrials)))
	events = np.vstack((events, np.ones(numTrials))).T.astype(int)

	return events

def makeEvents_rest(raw, crop_window, duration, Nevents):
	
	# Find the middle time point in the raw data
	# mid_time = raw.times[-1] / 2

	# # Define a time window around the middle time point
	# start = mid_time - (time_window / 2)
	# stop = mid_time + (time_window / 2)

	start = 0 + crop_window
	stop = raw.times[-1] - crop_window


	# Create events during the middle of the raw data
	events = mne.make_fixed_length_events(raw, start=start, stop=stop, duration=duration)

	# Randomly select N events
	np.random.shuffle(events)
	events = events[0:Nevents]

	# Sort the events by time
	events = np.sort(events, axis=0)

	return events	

def ampSpec(raw, n_fft):
	# Amplitude spectra 
	PSDs = raw.compute_psd(n_fft=n_fft, exclude='bads')
	ampSpecData = np.sqrt( PSDs.get_data() ) * 1e15 # in fT
	freq = PSDs.freqs

	return ampSpecData, freq

################################################
# Main Program

# Define lists to store evoked from all subjects
allEvokeds = []

# Loop through subjects, loading raw data and applying preprocessing
for s in subjects:

	if s<10:
		subjectID = 'mnsbp00' + str(s)
	else:
		subjectID = 'mnsbp0' + str(s)

	print(f'Starting {subjectID}...')

	# ID for the scaled-to-subject MRI folder	
	MRsubject = f'fsaverage_scaled_to_{subjectID}'

	# Define proc data directory for this subject
	subDataDir = os.path.join(dataDir, subjectID, 'meg')
	if not os.path.exists(subDataDir):
		os.makedirs(subDataDir)

	# Files to read
	raw_fname = os.path.join(subDataDir, rawPre + '-trans-raw.fif')

	# Files to write
	cleaned_raw_fname = os.path.join(subDataDir, cleaned_raw_fstem) 
	cleaned_epochs_fname = os.path.join(subDataDir, cleaned_epochs_fstem)
	cleaned_evoked_fname = os.path.join(subDataDir, cleaned_evoked_fstem)
	ica_fname = os.path.join(subDataDir, ica_fstem)
	raw_ampSpec_fname = os.path.join(subDataDir, raw_ampSpec_fstem)
	raw_cleaned_ampSpec_fname = os.path.join(subDataDir, raw_cleaned_ampSpec_fstem)
	ampSpec_freq_fname = os.path.join(subDataDir, ampSpec_freq_fstem)
	ICAName = os.path.join(dataDir, subjectID, 'meg', ica_fname)

	# Load the data
	raw_loc = mne.io.read_raw_fif(raw_fname, preload=True)

	# Define thresholds for event detection
	if rawPre == 'buttonPress':
		thresh_and_bads = event_thresholds_and_bads[subjectID]
		vThresh, badEvents = thresh_and_bads[0], thresh_and_bads[1]

	else:
		vThresh = 0.5
		badEvents = [] # There shouldn't be any bad events in the MNS tasks


	#################
	# Sensor Data Analysis

	# Get event timing
	if rawPre == 'buttonPress':
		events = getEvents_buttonPress(raw_loc, vThresh, badEvents)
	elif rawPre == 'rest':
		events = makeEvents_rest(raw_loc, crop_window, duration, Nevents)
		assert len(events) >= Nevents, f"Expected {Nevents} events, but got {len(events)}."
	else:
		events = getEvents_MNS(raw_loc, vThresh)
	
	raw_loc = raw_loc.copy().pick('mag')

	# Set the last three sensors as "bad" so they are:
	#	automatically excluded from HFC
	#	manually excluded from PSD
	#	included (but irrelevant) for reference array regression
	refChannels = raw_loc.info['ch_names'][-3::]
	raw_loc.info['bads'].extend(refChannels)

	# Filter and downsample
	raw_filt = raw_loc.copy()
	raw_filt = raw_filt.notch_filter(notch_freqs)
	raw_filt = raw_filt.filter(l_freq=l_freq1, h_freq=h_freq1)
	# raw_filt = raw_filt.filter(l_freq=l_freq1, h_freq=h_freq1, method='iir') # try different filter method

	# Grab the amplitude spectrum
	ampSpec_filt, freq = ampSpec(raw_filt.copy(), n_fft)

	# Drop channels with high noise at frequencies above 120 and below 150 Hz 
	#		(to be dropped)
	a = freq > 120
	b = freq < 145
	c = a*b
	hiFreqInd = np.where(c)[0]
	hiFreqAmp = np.mean(ampSpec_filt[:,hiFreqInd], axis=1)
	z_scores = np.abs(sstats.zscore(hiFreqAmp))
	outliers = np.where(z_scores > 2)[0]
	hiChans = [raw_filt.info['ch_names'][i] for i in outliers]
	# raw_filt.drop_channels(hiChans)
	# print('Dropped noisy channels...')
	# print(hiChans)
	raw_filt.info['bads'].extend(hiChans)   # mark bad channels as 'bad', instead of dropping them altogether. 

	# Append manually identified bad channels to the list of bad channels
	badChans = badChannels[subjectID]
	raw_filt.info['bads'].extend(badChans)
	print(raw_filt.info['bads'])

	# Rerun the amplitude spectrum without dropped channels
	ampSpec_filt, freq = ampSpec(raw_filt.copy().drop_channels(raw_filt.info['bads']), n_fft)

	# Get indices for sensors and references
	opmIndices = mne.channel_indices_by_type(raw_filt.info)['mag']
	numChannels = len(opmIndices)
	numSensors = numChannels-3
	sensorIndices = opmIndices[0:numSensors]
	referenceIndices = opmIndices[-3::]

	# Reference array regression 
	raw_regressed = referenceArrayRegression(raw_filt.copy(), opmIndices, sensorIndices, referenceIndices)

	# Homogenous field compensation (HFC)
	raw_hfc = raw_regressed.copy()
	projs = mne.preprocessing.compute_proj_hfc(raw_hfc.info, order=shcOrder)
	raw_hfc.add_proj(projs)
	raw_hfcApplied = raw_hfc.copy().apply_proj(verbose="error")

	# Grab the amplitude spectrum
	ampSpec_hfc, freq = ampSpec(raw_hfcApplied, n_fft)

	# Calculate the evoked responses (HFC projection is carried thru but not necessarily applied)
	epochs = mne.Epochs(raw_hfc, events, tmin=epochStart, tmax=epochEnd, 
		baseline=None, preload=True, proj=project)
	
	# Drop reference channels
	epochs.drop_channels(epochs.info['ch_names'][-3:])

	# ICA Artifact Decomposition (to be saved, not applied at this time)
	if not os.path.exists(ica_fname):
		reject = dict(mag=10e-12)
		picks = mne.pick_types(epochs.info, meg=True, eeg=False, eog=False,
							stim=False, exclude='bads')
		ica = ICA(n_components=0.99, method='fastica', random_state=42)
		ica.fit(epochs, picks=picks)

	# Apply baseline to epochs (after ICA, which is recommended by MNE)
	epochs.apply_baseline((baselineStart,baselineEnd))

	# Average across trials
	evoked = epochs.average().apply_baseline((baselineStart,baselineEnd)).crop(evokedStart,evokedEnd)

	allEvokeds.append(evoked)

	# Write preprocessed data to disk
	if writeOK:

		raw_hfcApplied.save(cleaned_raw_fname, overwrite=True) 
		epochs.save(cleaned_epochs_fname, overwrite=True) 
		if not os.path.exists(ica_fname):
			ica.save(ica_fname, overwrite=True)
		evoked.save(cleaned_evoked_fname, overwrite=True) 
		np.save(raw_ampSpec_fname, ampSpec_filt) 
		np.save(raw_cleaned_ampSpec_fname, ampSpec_hfc)
		np.save(ampSpec_freq_fname, freq) 

