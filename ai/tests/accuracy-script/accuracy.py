''' Usage: python3 accuracy.py [system-pred.out] [something.gold] '''

import sys

systemOutputfile = open(sys.argv[1], "r")
soLines = systemOutputfile.readlines()

goldFile = open(sys.argv[2], "r")
gfLines = goldFile.readlines()

correct = 0
total = 0


for i in range(len(soLines)):
    currPredLine = soLines[i].split(" ")
    currGoldLine = gfLines[i].split(" ")

    if len(currPredLine) == len(currGoldLine):
        for j in range(len(currPredLine)):
            if currPredLine[j] == currGoldLine[j]:
                correct += 1
            total += 1
    
    elif len(currPredLine) > len(currGoldLine):
        for j in range(len(currPredLine)):
            if j < len(currGoldLine) and currPredLine[j] == currGoldLine[j]:
                correct += 1
            total += 1
    else:
        for j in range(len(currGoldLine)):
            if j < len(currPredLine) and currPredLine[j] == currGoldLine[j]:
                correct += 1
            total += 1

print("Correct words are", correct)
print("Total words are", total)
print("Accuracy is:", ((correct/total)*100))

systemOutputfile.close()
goldFile.close()