# a file for the ALQ selector.

def lot_seize(lot):
    if lot <= 0:
        # raise ValueError("Lot must be a positive integer")
        None
    elif lot <= 500:
        return 32
    else:
        # raise ValueError("Bigger than 500 has not been implemented yet")
        None

def table_A(lotSize=500, inspectionLvl = 'I'):
    if lotSize >= 281 and lotSize <= 500 and inspectionLvl == 'I':
        code_letter = 'F'
        return code_letter
    else:
        return None
    
def table_B(codeLeter='F', inspection_lvl = 'I'):
    if codeLeter == 'F' and inspection_lvl == 'I':
        return 32
    else:
        return None

def rejectionValue(codeLeter, acceptablePercent):
    sample_sicze = 0
    reject_nr = 0
    
    if codeLeter == 'F' and acceptablePercent <= 0.4:
        sample_size = 32
        reject_nr = 1
    elif codeLeter == 'F' and acceptablePercent <= 6.5:
        sample_size = 20
        reject_nr = 2
    elif codeLeter == 'F' and acceptablePercent <= 15:
        sample_size = 20
        reject_nr = 6
    else:
        sample_size = None
        reject_nr = None
    
    values = sample_size,reject_nr
    return values

def AQL_selector(lotSize, inspectionLvl, acceptablePercent):
    code_letter = table_A(lotSize, inspectionLvl)
    rejection_value = rejectionValue(code_letter, acceptablePercent)
    return rejection_value    

def AQL_classificationmaker(lotSize, inspectionLvl, acceptablePercentages):
    classDict = dict()
    rejectDict = dict()
    classCounter = 0
    for i in acceptablePercentages:
        classCounter += 1
        classDict[classCounter] = AQL_selector(lotSize, inspectionLvl, i)
        sample_size = max(classDict.values(), key=lambda x: x[0])[0]
        rejectDict[classCounter] = classDict[classCounter][1]

    classifier = sample_size, rejectDict
    return classifier

def AQL_classification(batchList, classification):
    sampleSizeCheck = len(batchList) - classification[0]
    if sampleSizeCheck != 0:
        raise ValueError("The batchList and the sampleSize do not match")
    reject = sum(batchList)
    print(reject)   
    for i in classification[1]:
            if reject >= classification[0][i].values():
                return f'This lot is class_{classification[0][i].keys()}'
    
    return f'This lot is rejected'

    

def testfunction():
    print('testing!')

    # lot_seizeList = [501, 500, 499, 281, 280, 279, 0, -1, -500, 1000]
    # lot_seizeCounter = 0
    # for i in lot_seizeList:
    #     lot_seizeCounter += 1
    #     lot_seize_result = lot_seize(i)
    #     print(f'lot_seizeList{lot_seizeCounter} ; {lot_seize_result}')

    # tableTestTupleList = [(500, 'I'), (500, 'II'), (500, 'III'), (499, 'I'), (501, 'I'), (281, 'I'), (280, 'I'), (279, 'I')]
    # tableTestCounter = 0
    # for i in tableTestTupleList:
    #     tableTestCounter += 1
    #     table_A_result = table_A(i[0], i[1])
    #     print(f'tableTestTupleList{tableTestCounter} ; {table_A_result}')

    # rejectionValueTupleList = [('F', 0.3), ('F', 0.4), ('F', 0.5), ('F', 6.5), ('F', 6.6), ('F', 7), ('F', 15), ('F', 16), ('F', 100)]
    # rejectionValueCounter = 0
    # for i in rejectionValueTupleList:
    #     rejectionValueCounter += 1
    #     rejectionValue(i[0], i[1])
    #     print(f'rejectionValueTupleList{rejectionValueCounter} ; {i}')

    # AQL_selectorTupleList = [(500, 'I', 0.3)  # , (500, 'II', 0.3),(500, 'I', 0.4), (500, 'I', 0.5), (500, 'I', 6.5), (500, 'I', 6.6), (500, 'I', 7), (500, 'I', 15), (500, 'I', 16), (500, 'I', 100)
    #                          ]
    # AQL_selectorCounter = 0
    # for i in AQL_selectorTupleList:
    #     AQL_selectorCounter += 1
    #     AQL_selector_result = AQL_classificationmaker(i[0], i[1], i[2])
    #     print(f'AQL_selectorTupleList{AQL_selectorCounter} ; {AQL_selector_result}')
    
    AQL_classifierTupleList = [(500, 'I', [0.4, 6.5, 15])]
    # make a list of 32 1's and 0's to test the AQL_classifier
    testlist = []
    classification = AQL_classificationmaker( AQL_classifierTupleList[0][0], AQL_classifierTupleList[0][1], AQL_classifierTupleList[0][2])
    # print(classification)
    
    # .make a list of the classes in a dictionary

    # print(classification[1].keys())
    # print(classification[1].values())
    
    sampleTestValueList = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Klasse = AQL_classification(sampleTestValueList, classification)
    print (Klasse)

  

##### fix last part! #####
    
 


if __name__ == '__main__':
    testfunction()


