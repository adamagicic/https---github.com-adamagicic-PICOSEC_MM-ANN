import numpy as np

def loadingBar(p=0, w = 30):
    left = int(p*w)
    right = int(w - left)
    print('\r[','#'*left,'-'*right,'] ',f'{p*100:.0f}%',flush=True,sep='',end=' ')

def loadmatFromOctave(matFileData, **opts):
    events = []
    if opts['MCP'] == 1:
        detectorData = matFileData['MCP_data']
        print("Converting MCP_data structure into python dict...")
        for i in range(len(detectorData[0])):
            event = detectorData[0][i]
            eventDict = {'fail': event[0][0,0],
                         'sig': {'length': event[1][0,0][0][0,0], 'max': {'idx': event[1][0,0][1][0,0][1][0,0], 'y': event[1][0,0][1][0,0][0][0,0]},
                                                                    'min': {'idx': event[1][0,0][2][0,0][1][0,0], 'y': event[1][0,0][2][0,0][0][0,0]},
                                    'blrms': event[1][0,0][3][0,0], 'blavg': event[1][0,0][4][0,0],
                                    'startpoint': {'idx': event[1][0,0][5][0,0][0][0,0], 'x': event[1][0,0][5][0,0][1][0,0], 'y': event[1][0,0][5][0,0][2][0,0]},
                                    'half_up': {'idx': event[1][0,0][6][0,0][0][0,0], 'x': event[1][0,0][6][0,0][1][0,0], 'y': event[1][0,0][6][0,0][2][0,0]},
                                    'endpoint': {'idx': event[1][0,0][7][0,0][0][0,0], 'x': event[1][0,0][7][0,0][1][0,0], 'y': event[1][0,0][7][0,0][2][0,0]},
                                    'half_dn': {'idx': event[1][0,0][8][0,0][0][0,0], 'x': event[1][0,0][8][0,0][1][0,0], 'y': event[1][0,0][8][0,0][2][0,0]},
                                    't_half': event[1][0,0][9][0,0],
                                    'e_peak_end': {'idx': event[1][0,0][10][0,0][1][0,0], 'x': event[1][0,0][10][0,0][2][0,0], 'y': event[1][0,0][10][0,0][0][0,0]},
                                    'k_V2C': event[1][0,0][11][0,0],
                                    'charge': {'lead_edge': event[1][0,0][12][0,0][0][0,0], 'e_peak': event[1][0,0][12][0,0][1][0,0], 'all': event[1][0,0][12][0,0][2][0,0]},},
                        'sigmoid': {'start': event[2][0,0][0][0,0], 'end': event[2][0,0][1][0,0], 'npoints': event[2][0,0][2][0,0], 'p': event[2][0,0][3][0],
                                    'err': event[2][0,0][4][0], 'chi': event[2][0,0][5][0,0], 'time20': event[2][0,0][6][0,0], 'sevals': event[2][0,0][7][0,0], 'timepoint': event[2][0,0][8][0,0]},
                        'event_id': event[3][0,0]}
            events.append(eventDict)
            loadingBar((i+1)/len(detectorData[0]))
        matFileData.update({'MCP_data': np.array(events)})
        print('')
    else:
        detectorData = matFileData['MM_data']
        print("Converting MM_data structure into python dict...")
        for i in range(len(detectorData[0])):
            event = detectorData[0][i]
            eventDict = {'fail': event[0][0,0],
                         'sig': {'length': event[1][0,0][0][0,0], 'max': {'idx': event[1][0,0][1][0,0][1][0,0], 'y': event[1][0,0][1][0,0][0][0,0]},
                                                                    'min': {'idx': event[1][0,0][2][0,0][1][0,0], 'y': event[1][0,0][2][0,0][0][0,0]},
                                    'blrms': event[1][0,0][3][0,0], 'blavg': event[1][0,0][4][0,0],
                                    'startpoint': {'idx': event[1][0,0][5][0,0][0][0,0], 'x': event[1][0,0][5][0,0][1][0,0], 'y': event[1][0,0][5][0,0][2][0,0]},
                                    'half_up': {'idx': event[1][0,0][6][0,0][0][0,0], 'x': event[1][0,0][6][0,0][1][0,0], 'y': event[1][0,0][6][0,0][2][0,0]},
                                    'endpoint': {'idx': event[1][0,0][7][0,0][0][0,0], 'x': event[1][0,0][7][0,0][1][0,0], 'y': event[1][0,0][7][0,0][2][0,0]},
                                    'half_dn': {'idx': event[1][0,0][8][0,0][0][0,0], 'x': event[1][0,0][8][0,0][1][0,0], 'y': event[1][0,0][8][0,0][2][0,0]},
                                    't_half': event[1][0,0][9][0,0],
                                    'e_peak_end': {'idx': event[1][0,0][10][0,0][1][0,0], 'x': event[1][0,0][10][0,0][2][0,0], 'y': event[1][0,0][10][0,0][0][0,0]},
                                    'k_V2C': event[1][0,0][11][0,0],
                                    'charge': {'lead_edge': event[1][0,0][12][0,0][0][0,0], 'e_peak': event[1][0,0][12][0,0][1][0,0], 'all': event[1][0,0][12][0,0][2][0,0]},},
                        'sigmoid': {'start': event[2][0,0][0][0,0], 'end': event[2][0,0][1][0,0], 'npoints': event[2][0,0][2][0,0], 'p': event[2][0,0][3][0],
                                    'err': event[2][0,0][4][0], 'chi': event[2][0,0][5][0,0], 'time20': event[2][0,0][6][0,0], 'sevals': event[2][0,0][7][0,0], 'timepoint': event[2][0,0][8][0,0]},
                        'event_id': event[3][0,0], 'x': event[4][0,0], 'y': event[5][0,0]}
            events.append(eventDict)
            loadingBar((i+1)/len(detectorData[0]))
        matFileData.update({'MM_data': np.array(events)})
        print('')
    return np.array(events)