import lecroyscope as ls
import numpy as np
import pandas as pd
import time
import os

def loadingBar(p=0, w = 30):
    left = int(p*w)
    right = int(w - left)
    print('\r[','#'*left,'-'*right,'] ',f'{p*100:.0f}%',flush=True,sep='',end=' ')

def readData(path, CH, N):
    "Reads scope data (from channel CH) from N files in path, stores them in one numpy array; filename format CH--Trace--00000.trc"
    totData = []
    print(f"Loading channel %d data from %d files..." % (CH, N))
    t0 = time.time()
    for n in range(N):
        filename = f"C%d--Trace--%05d.trc" % (CH, n)
        data = ls.Trace(path/filename)
        for i in range(len(data.y)):
            totData.append(data.y[i])
        loadingBar((n+1)/N)
    t1 = time.time()
    print(f"Elapsed time: %lf s" % (t1-t0))
    return np.array(totData)

def dataCutSelection(matData, tracker = 0):
    "Performs selection for data cut."
    threshholdV = 0.02

    eventsToRemove = []
    for i in range(len(matData['MM_data'])):

        # failed fits
        if matData['MM_data'][i]['sigmoid']['timepoint'] != matData['MM_data'][i]['sigmoid']['timepoint']:
            eventsToRemove.append(i)
            continue

        # amplitude cut
        if matData['MM_maxy'][0,i] < threshholdV:
            eventsToRemove.append(i)
            continue
        
        # unreasonable SAT cut
        if matData['time_diff_sigmoid'][0,i] > -1 or matData['time_diff_sigmoid'][0,i] < -2:
            eventsToRemove.append(i)
            continue

        # events with no tracker data
        if tracker != 0:
            if matData['trackerX'][0,i] < 1e-10 or matData['trackerY'][0,i] < 1e-10:
                eventsToRemove.append(i)
                continue
        
        # events with weak signal
        if matData['MM_maxy'][0,i] < matData['MM_data'][i]['sig']['blrms']:
            eventsToRemove.apend(i)
            continue

    return np.array(eventsToRemove)

def dataCut(matData, dataSelection):
    "Cuts data specified in in dataSelection."
    initL = len(matData['MM_data'])
    print(f'Removing %d events...' % len(dataSelection))
    t0 = time.time()
    matData.update({'MM_data': np.delete(matData['MM_data'], dataSelection)})
    matData.update({'MCP_data': np.delete(matData['MCP_data'], dataSelection)})
    matData.update({'time_diff_sigmoid': np.delete(matData['time_diff_sigmoid'][0], dataSelection)})
    matData.update({'MCP_maxy': np.delete(matData['MCP_maxy'][0], dataSelection)})
    matData.update({'MM_maxy': np.delete(matData['MM_maxy'][0], dataSelection)})
    matData.update({'trackerX': np.delete(matData['trackerX'][0], dataSelection)})
    matData.update({'trackerY': np.delete(matData['trackerY'][0], dataSelection)})
    t1 = time.time()
    print(f'Elapsed time: %lf s' % (t1-t0))
    if (len(matData['MM_data']) == initL-len(dataSelection)):
        print(f'Successfully removed events. Number of remaining events: %d' % len(matData['MM_data']))
        return 0
    else:
        print('Event removal failed, final number of events does not match dataSelection length!')
        return 1

def choosePointsfromMAT(matData, signalData):
    "Chooses points to write into CSV file."
    pointsToWrite = []
    t0 = time.time()
    print('Choosing signal points...')
    for i in range(len(matData['MM_data'])):
        # amplitude = np.min(signalData[i])
        startpoint = np.argmin(signalData[i])
        #while (signalData[i][startpoint] > 0.3*amplitude or signalData[i][startpoint+1] > 0.3*amplitude or signalData[i][startpoint+2] > 0.3*amplitude) and startpoint < len(signalData[i])-16:
        #    startpoint = startpoint + 1
        # startpoint = int(matData['MM_data'][i]['sig']['max']['idx'])
        if startpoint > 56 and startpoint < 9994:
            chosenPoints = signalData[i][startpoint-56:startpoint+8]
        else:
            chosenPoints = np.zeros(64)
        pointsToWrite.append(chosenPoints)
        loadingBar((i+1)/len(matData['MM_data']))
    print('')
    t1 = time.time()
    print(f'Elapsed time: %lf s' % (t1-t0))
    return pointsToWrite

def writeData(pointSignalData, matDataLabels, saveName, event0, eventF):
    "Writes the choosePoints data to .csv file for chosen events"
    print('Creating pandas dataframe...')
    t0 = time.time()
    df = pd.DataFrame(np.arange(eventF-event0), columns=['eventNo'])
    df = pd.concat([df, pd.Series(matDataLabels[event0:eventF])], axis = 1)
    df['signal_data'] = pointSignalData
    df.columns = ['eventNo', 'time_label', 'signal_data']
    print("Writing dataframe to .csv file...")
    df.to_csv(saveName, encoding = 'utf-8', index = False)
    print(f"Finished! File written to ~\%s" % saveName)
    t1 = time.time()
    print(f'Elapsed time: %lf s' % (t1-t0))

    return df