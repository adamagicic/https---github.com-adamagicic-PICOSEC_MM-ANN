# preparing .csv file for tf dataset
from functions.rwFunctions import *
from scipy.io import loadmat
from functions.readMATfile import loadmatFromOctave
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt

runID = "353"

signalDataPath = Path("Q:\ANN_Vuko_Adam\PICOSEC_scope_tracker_DATA\Pool2") / ("Run" + runID)
matPath = Path("Q:\ANN_Vuko_Adam\Timing_tracker_octave_WIN") / ("Run" + runID + "MATfile.mat")

# reading signal data from .trc files
N = int(len(os.listdir(signalDataPath))/4)
signalDataMM = readData(signalDataPath, 4, N)
signalDataMCP = readData(signalDataPath, 1, N)

# reading analyzed data from .mat file
print("Loading .mat file...")
t0 = time.time()
matData = loadmat(matPath)
t1 = time.time()
print(f"Elapsed time: %lf s" % (t1-t0))

# converting Octave data structures from .mat file into python dicts
matDataMM = loadmatFromOctave(matData, MCP = 0)
matDataMCP = loadmatFromOctave(matData, MCP = 1)

cut = dataCutSelection(matData)
cutData = dataCut(matData, cut)

chosenPoints = choosePointsfromMAT(matData, signalDataMM)
writeData(chosenPoints, matData['time_diff_sigmoid'], 'CSV\Run351.csv', 0, len(matData['time_diff_sigmoid']))

# upitni događaji: 10 008, 27 841