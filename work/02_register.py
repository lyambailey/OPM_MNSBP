'''
Outline:
The purpose of this script is to scale and align a standard head model to a subject's 
head and sensor positions

Each time we run the script, we'll see an alignment of the standard with the subject's 
head and sensor positions. Alignment parameters for each subject are read out of 
manualRegistrationParams.xlsx. This needs to be updated (and the script re-run) until
we find parameters for optimal fit. 

Note that this script handles data from one subject at a time!
'''

# Import packages
import open3d as o3d
import mne
import mne.transforms as mt
import numpy as np
import pandas as pd
import os
import copy
from scipy.spatial.transform import Rotation as R

################################################
# User input
subjectID = 'mnsbp011'
rawPre = 'MNS_longISI'

# Make plots?
plotOK = False

# Show head points in plots?
showHeadPoints = True

# Write data?
writeOK = True

# Tweak registration?
tweak = True

################################################
# Other variables

MRsubject = 'fsaverage'
MRsubjectScaled = f'{MRsubject}_scaled_to_{subjectID}'

# Main directories
projectDir = os.path.join('C:', os.sep, 'Users', 't_bar', 'Documents', 'Data', 'MNSandBP')
dataDir = os.path.join(projectDir, 'raw_data')
outputDir = os.path.join(projectDir, 'proc_data')
resultsDir = os.path.join(projectDir, 'results')
supportDir = os.path.join(projectDir, 'supportFiles')
subjectsDir = os.path.join(projectDir, 'subjects')

# Files to read
rawDigiFif = os.path.join(outputDir, subjectID, 'meg', rawPre + '-digi-raw.fif')
regiMLP = os.path.join(outputDir, subjectID, 'digi', 'registration.mlp')
rotatedFidPly = os.path.join(outputDir, subjectID, 'digi', 'fiducials_headCoords.ply')
transHeadPly = os.path.join(outputDir, subjectID, 'digi', 'headScan_headCoords.ply')
manRegParamsxl = os.path.join(supportDir, 'manualRegistrationParams.xlsx')

# Files to write
rawRegFif = os.path.join(projectDir, 'proc_data', subjectID, 'meg', rawPre + '-trans-raw.fif')
headShapeMontageFif = os.path.join(projectDir, 'proc_data', subjectID, 'digi', 'headShape_fids.fif')
MRItransFifName = os.path.join(projectDir, 'proc_data', subjectID, 'digi', 'head_mri-trans.fif')
MEGtransFifName = os.path.join(projectDir, 'proc_data', subjectID, 'digi', 'head_meg-trans.fif')

################################################
# Registration functions

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

def read_MeshLab_trans(regMeshLabFile, lineNum):

	with open(regMeshLabFile) as file:
		lines = [line.rstrip() for line in file]
	transLine = lines[lineNum]
	# Split sentence into list of string values 
	transList = transLine.split()
	# Turn strings into floats and make array
	transMatrix = np.asarray([float(i) for i in transList]).reshape(4,4)
	# Change the translation into metres
	transMatrix[0:3,3] = transMatrix[0:3,3]*1e-3
	# Place in MNE variable
	trans = mt.Transform(fro="meg", to="head", trans=transMatrix)

	return trans

def set_head_coords(fidPly, headPly):

	# Read in files
	fiducials = o3d.io.read_point_cloud(fidPly)
	head = o3d.io.read_point_cloud(headPly)

	# Find clusters of points in the fiducials data. There should be three
	with o3d.utility.VerbosityContextManager(
			o3d.utility.VerbosityLevel.Debug) as cm:
		labels = np.array(
			fiducials.cluster_dbscan(eps=2, min_points=10, print_progress=True))

	# Push on if there are three clusters
	if labels.max() == 2:
		
		CofMs = []
		# Find the centre of mass of each cluster
		for fidNum in range(3):
			
			# Find which points relate to this cluster
			fidIndex = np.where(labels==fidNum)[0]
			# Get the points that relate to this cluster 
			fidPts = np.asarray([np.asarray(fiducials.points)[x,:] for x in fidIndex])
			# Get the centre of mass of those points 
			fidCofM = np.mean(fidPts, axis=0)
			
			# Compile centre of masses across loops
			CofMs.append( fidCofM )
	CofMs = np.asarray(CofMs)

	# Nasion has the smallest z value
	nasInd = np.argmin(CofMs[:,2])
	NAS = CofMs[nasInd,:]

	# Left PA has the largest y value
	leftInd = np.argmax(CofMs[:,1])
	LPA = CofMs[leftInd,:]

	# Right PA has the smallest y value
	rightInd = np.argmin(CofMs[:,1])
	RPA = CofMs[rightInd,:]

	# Reorder the points
	fiducialPts = np.vstack((NAS, np.vstack((LPA, RPA))))

	# Define the head coordinate frame
	Xslope = RPA-LPA
	xhat = Xslope/np.sqrt(np.dot(Xslope,Xslope))
	t_numer = xhat[0]*(NAS[0]-LPA[0]) + xhat[1]*(NAS[1]-LPA[1]) + xhat[2]*(NAS[2]-LPA[2]) 
	t_denom= np.dot(xhat,xhat)
	t = t_numer/t_denom
	origin = LPA + t*xhat
	Yslope = NAS-origin
	yhat = Yslope/np.sqrt(np.dot(Yslope,Yslope))
	zhat = np.cross(xhat,yhat)

	# Define the rotation matrix for the new coordinate frame
	rotMatrix = np.hstack((xhat,0))
	rotMatrix = np.vstack((rotMatrix, np.hstack((yhat,0))))
	rotMatrix = np.vstack((rotMatrix, np.hstack((zhat,0))))
	rotMatrix = np.vstack((rotMatrix, np.asarray([0,0,0,1])))

	# Translate the fiducials to the new origin
	fid_translate = copy.deepcopy(fiducials).translate(-origin)


	# Rotate and save the translated fiducials
	fiducials_headCoords = copy.deepcopy(fid_translate).transform(rotMatrix)

	# Translate, rotate and save the head scan in the head coordinates
	head_headCoords = copy.deepcopy(head).translate(-origin).transform(rotMatrix)

	return rotMatrix, origin, fiducials_headCoords, head_headCoords

def extract_fids(fiducials_headCoords):

	# Find clusters of points in the fiducials in head coords
	with o3d.utility.VerbosityContextManager(
			o3d.utility.VerbosityLevel.Debug) as cm:
		labels = np.array(
			fiducials_headCoords.cluster_dbscan(eps=2, min_points=10, print_progress=True))
	if labels.max() == 2:
		CofMs = []
		# Find the centre of mass of each cluster
		for fidNum in range(3):
			# Find which points relate to this cluster
			fidIndex = np.where(labels==fidNum)[0]
			# Get the points that relate to this cluster 
			fidPts = np.asarray([np.asarray(fiducials_headCoords.points)[x,:] for x in fidIndex])
			# Get the centre of mass of those points 
			fidCofM = np.mean(fidPts, axis=0)
			# Compile centre of masses across loops
			CofMs.append( fidCofM )
	CofMs = np.asarray(CofMs)*1e-3
	# Nasion has the largest y value
	nasInd = np.argmax(CofMs[:,1])
	NAS = CofMs[nasInd,:]
	# Left PA has the smallest x value
	leftInd = np.argmin(CofMs[:,0])
	LPA = CofMs[leftInd,:]
	# Right PA has the largest xvalue
	rightInd = np.argmax(CofMs[:,0])
	RPA = CofMs[rightInd,:]

	return NAS, RPA, LPA

def downsampled_shape(head_headCoords):

	# Make a heavily downsampled (decimated) head for MEG-MRI registration
	headDecim = copy.deepcopy(head_headCoords).voxel_down_sample(voxel_size=10)
	headShape = np.asarray(headDecim.points)*1e-3   # Now in metres

	# Extract expanded racoon mask only
	# Keep only points with z > -0.03
	#a = np.where(headShape[:,2]>-0.03)[0]
	#headShape = headShape[a,:]
	# Keep only points with z < 0.07
	#a = np.where(headShape[:,2]<0.07)[0]
	#headShape = headShape[a,:]
	# Keep only points with y > 0.03m
	#a = np.where(headShape[:,1]>0.03)[0]
	#headShape = headShape[a,:]

	return headShape

def auto_headMR_registration(info, MRsubject, subjects_dir):

	# Set up initial registration (no rotation or translation applied)
	fiducials = "auto"  # get fiducials I defined from the bem folder
	coreg = mne.coreg.Coregistration(info, MRsubject, subjects_dir, fiducials=fiducials)	

	# Fit to the fiducials (MEG and MRI)
	coreg.fit_fiducials(verbose=True)

	# Initial fit to the headshape using ICP
	coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)

	# Drop any points that are far from the head surface
	coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

	# New - scale coreg
	# coreg = coreg.set_scale_mode('3-axis').set_scale(scale)

	# Refit to the ICP with reduced point set
	coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

	dists = coreg.compute_dig_mri_distances() * 1e3  # in mm

	print(
		f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
		f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
	)

	return coreg

def modify_rotation(trans_matrix, rotation_values):
    """
    Modifies the rotation matrix in the given transformation matrix.
   
    Parameters:
        trans_matrix (np.ndarray): 4x4 transformation matrix.
        rotation_values (tuple): Rotation increments (in radians) for (x, y, z) axes.
       
    Returns:
        np.ndarray: Updated transformation matrix.
    """
    # Extract the current rotation matrix
    current_rotation = R.from_matrix(trans_matrix[:3, :3])

    # Create a rotation object for the incremental rotations
    delta_rotation = R.from_euler('xyz', rotation_values, degrees=False)

    # Combine the current rotation with the incremental rotation
    # The new rotation is applied first
    new_rotation = delta_rotation * current_rotation

    # Update the rotation matrix in the transformation matrix
    trans_matrix[:3, :3] = new_rotation.as_matrix()
   
    return trans_matrix

def tweak_headMR_registration(coreg, manualTrans, manualRot):

	manTransMatrix = copy.deepcopy(coreg).trans.get('trans')
	manTransMatrix[0:3,3] = manTransMatrix[0:3,3] + manualTrans

	manTransMatrix = modify_rotation(manTransMatrix, manualRot)

	manTrans = mne.transforms.Transform(fro="head", to="mri", trans=manTransMatrix)

	return manTrans

def get_manual_registration_parameters(manRegParamsxl):

	# Read reg parameters from excel sheet
	manRegParams = pd.read_excel(manRegParamsxl)

	# Get the row for this subject
	manRegParams = manRegParams.loc[manRegParams['subjectID'] == subjectID]

	# Assign parameters to appropriate variables
	MRscale = float(manRegParams['scale'].iloc[0])
	manualTranslation = tuple(map(tuple, manRegParams[['x_trans','y_trans','z_trans']].values))[0]
	manualRotation = tuple(map(tuple, manRegParams[['x_rot','y_rot','z_rot']].values))[0]

	return manRegParams, MRscale, manualTranslation, manualRotation

################################################
# Main Program

print(subjectID)

# Scale MR to fit subject's head
manRegParams, MRscale, manualTranslation, manualRotation = get_manual_registration_parameters(manRegParamsxl)

mne.scale_mri('fsaverage', MRsubjectScaled, scale = MRscale, subjects_dir = subjectsDir, overwrite=True)

# Load the raw data
raw = mne.io.read_raw_fif(rawDigiFif, preload=True)
raw_meg = raw.copy()#.pick('mag')
raw_loc = raw.copy() 

# Extract head to meg transform from Meshlab registration project file
transMEG = read_MeshLab_trans(regiMLP, 17)

# Add head->meg transform to raw variable
raw_loc.info['dev_head_t'] = transMEG

# Automatically register head shape to MR
coreg = auto_headMR_registration(raw_loc.info, MRsubjectScaled, subjectsDir)

transMRI = copy.deepcopy(coreg).trans #.get('trans')  # .get('trans') returns a numpy array, which does not play nice with mne.write_trans (ln 317)

###################
# Manually translate the MR to adjust the head-MRI registration
if tweak:
	transMRI = tweak_headMR_registration(coreg, manualTranslation, manualRotation)


###########
# Write output files
if writeOK:
	# Make the folder  for processed files
	if not os.path.exists(os.path.join(outputDir, subjectID, 'meg')):
		os.makedirs(os.path.join(outputDir, subjectID, 'meg'))
	if not os.path.exists(os.path.join(outputDir, subjectID, 'digi')):
		os.makedirs(os.path.join(outputDir, subjectID, 'digi'))

	# Save head to MRI transform
	mne.write_trans(MRItransFifName, transMRI, overwrite=True)
	# Save head to MEG transform
	mne.write_trans(MEGtransFifName, transMEG, overwrite=True)
	# Save raw data with the head to MEG transform included
	raw_loc.save(rawRegFif, overwrite=True)

############
# Make plots
if plotOK:
	fig = mne.viz.plot_alignment(raw_loc.info, trans=transMRI, **dict(
		subject=MRsubjectScaled,
		subjects_dir=subjectsDir,
		surfaces="head",
		dig=showHeadPoints,
		eeg=[],
		meg="sensors",
		show_axes=True,
		coord_frame="meg",
	))
	mne.viz.set_3d_view(fig, **dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0)))



