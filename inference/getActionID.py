import sys 
import json
sys.path.append('..')

from inference.table import actions

def LoadSeqJsonDict(rootPath,subject):
    # load the mapping dict between the encoder seq and the video 
    jsonFilePath = '{}/S{}/S{}.json'.format(rootPath,subject,subject)
    print('jsonFilePath',jsonFilePath)
    with open(jsonFilePath,'r') as f:
        vedioJsonDict = json.load(f)

    return vedioJsonDict

def getActionID(camPara,jsonDict,debug=False):
    # mapping the 'S_SeqID_CameraID_ActionID' to the raw video name 
    valueName = 'S_' + str(camPara[0]) + '_' + 'C_' + str(camPara[1]) + '_' +  str(camPara[3])

    actionName = None
    for k,v in jsonDict.items():
        if valueName in v and v in valueName:
            actionName = k
    rawVedioName = actionName

    actionName = actionName.replace('WalkingDog','WalkDog')
    act = 0
    for action in actions:
        act +=1
        if action in actionName:
            break
    if 'Sitting' in actionName and 'SittingDown' not in actionName:
        act = 9
    if 'Sitting' in actionName and 'SittingDown'  in actionName:
        act = 10
    #"WalkDog","Walking"
    if 'Walking' in actionName and 'WalkingDog' not in actionName:
        act = 14
    if 'Walking' in actionName and 'WalkingDog'  in actionName:
        act = 13
    action = actions[act - 1]
    if  debug:
        print ('action',action)
        print ('rawVedioName',rawVedioName)

    return act - 1,rawVedioName