"""
Usage: python3 generate-report.py [source] [initial-output.out] [initial-output.gold] 
        [intermediate-output.out] [intermediate-output.gold] [final-output.out] 
        [final-output.gold] [report.txt]
"""
import sys
import math

def calcIncorrectPlusToken(system, gold):
    if "[+]" in system:
        count = 0
        system = system.strip().split(" ")
        gold = gold.strip().split(" ")
        for i in range(len(system)):
            if "[+]" in system[i] and "[+]" not in gold[i]:
                count += 1
        return count
    else:
        return 0

def calcMissingPlusToken(system, gold):
    if "[+]" in gold:
        count = 0
        system = system.strip().split(" ")
        gold = gold.strip().split(" ")
        for i in range(len(system)):
            if "[+]" in gold[i] and "[+]" not in system[i]:
                count += 1
        return count
    else:
        return 0

def calcIncorrectMinusToken(system, gold):
    if "[-]" in system:
        count = 0
        system = system.strip().split(" ")
        gold = gold.strip().split(" ")
        for i in range(len(system)):
            if "[-]" in system[i] and "[-]" not in gold[i]:
                count += 1
        return count
    else:
        return 0

def calcMissingMinusToken(system, gold):
    if "[-]" in gold:
        count = 0
        system = system.strip().split(" ")
        gold = gold.strip().split(" ")
        for i in range(len(system)):
            if "[-]" in gold[i] and "[-]" not in system[i]:
                count += 1
        return count
    else:
        return 0

def overallAccuracy(system, gold):
    correct = 0
    total = 0

    for i in range(len(system)):
        currPredLine = system[i].split(" ")
        currGoldLine = gold[i].split(" ")

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

    return str(round((correct/total) * 100, 1))

def accuracy(system, gold):
    system = system.strip().split(" ")
    gold = gold.strip().split(" ")
    
    total = 0
    correct = 0
    for i in range(len(system)):
        if system[i] == gold[i]:
            correct += 1
        total += 1

    return str(round((correct/total) * 100, 1))

globalHashtagTotal = 0
globalHashtagCorrect = 0
def hashtagAccuracy(system, gold):
    global globalHashtagTotal
    global globalHashtagCorrect

    system = system.strip().split(" ")
    gold = gold.strip().split(" ")

    total = 0
    correct = 0

    for i in range(len(system)):
        if gold[i] == "#":
            if system[i] == "#":
                correct += 1
                globalHashtagCorrect += 1
            total += 1
            globalHashtagTotal += 1

    return str(round((correct/total) * 100, 1))

def checkHashtagAlignmentFromSystem(system, gold):
    system = system.strip().split(" ")
    gold = gold.strip().split(" ")

    count = 0
    for i in range(len(gold)):
        if i < len(system) and system[i] == "#" and gold[i] != "#":
            count += 1
    
    return str(count)

def checkHashtagGenerationFailure(system, gold):
    system = system.strip().split(" ")
    gold = gold.strip().split(" ")

    count = 0
    for i in range(len(gold)):
        if i < len(system) and gold[i] == "#" and system[i] != "#":
            count += 1

    return str(count)

def prettyBins(dict):
    listOfAccuracies = list(dict.keys())
    listOfAccuracies.sort()

    for acc in listOfAccuracies:
        print(("|" + str(acc) + "%|" + "\t").expandtabs(10), end = "")
    print()

    for acc in listOfAccuracies:
        print(("|" + str(dict[acc]) + "|" + "\t").expandtabs(10), end = "")
    print()

def lineBreakdown(ia, ig, inta, intg, groupsOf):
    totalLines = len(ia)
    numberOfGroupsOfThousands = totalLines // groupsOf

    count = 0
    for _ in range(numberOfGroupsOfThousands):
        print("Accuracy of {}-{} lines: Initial {}, Source Aligned {}"
        .format(count, count + groupsOf, overallAccuracy(ia[count:count+groupsOf+1], ig[count:count+groupsOf+1]),
        overallAccuracy(inta[count:count+groupsOf+1], intg[count:count+groupsOf+1])))

        count += groupsOf

    # Checking any lines that are left
    print("Accuracy of {}-{} lines: Initial {}, Source Aligned {}"
        .format(count, totalLines, overallAccuracy(ia[count:totalLines], ig[count:totalLines]),
        overallAccuracy(inta[count:totalLines], intg[count:totalLines])))
    

source = open(sys.argv[1], "r").readlines()
initialArabizi = open(sys.argv[2], "r").readlines()
initialGold = open(sys.argv[3], "r").readlines()
intermediateArabizi = open(sys.argv[4], "r").readlines()
intermediateGold = open(sys.argv[5], "r").readlines()
finalArabizi = open(sys.argv[6], "r").readlines()
finalGold = open(sys.argv[7], "r").readlines()

report = open(sys.argv[8], "w")

incorrectHashtag = 0
hashtagFailure = 0

incorrectInitialAlignment = 0
incorrectIntermediateAlignment = 0
incorrectFinalAlignment = 0

incorrectPlusToken = 0
missingPlusToken = 0
incorrectMinusToken = 0
missingMinusToken = 0

initialAccuracyBin = {}
intermediateAccuracyBin = {}

for i in range(len(source)):
    report.write(("Source: \t" + source[i]).expandtabs(30))
    report.write(("InitialOut: \t" + initialArabizi[i]).expandtabs(30))
    report.write(("InitialGold: \t" + initialGold[i]).expandtabs(30))
    
    report.write(("AlignedFinalOut: \t" + intermediateArabizi[i]).expandtabs(30))
    report.write(("AlignedFinalGold: \t" + intermediateGold[i]).expandtabs(30))
    
    report.write(("UnalignedFinalOut: \t" + finalArabizi[i]).expandtabs(30))
    report.write(("UnalignedFinalGold: \t" + finalGold[i]).expandtabs(30))

    if "#" in initialArabizi[i]:
        hashtagMisalignmentCount = checkHashtagAlignmentFromSystem(initialArabizi[i], initialGold[i])
        if hashtagMisalignmentCount != "0":
            report.write("ERROR: Incorrect hashtag in " + hashtagMisalignmentCount + " place(s)\n")
            incorrectHashtag += 1

    if "#" in initialGold[i]:
        hashtagGenerationFailureCount = checkHashtagGenerationFailure(initialArabizi[i], initialGold[i])
        if hashtagGenerationFailureCount != "0":
            report.write("ERROR: Failed to generate hashtag in " + hashtagGenerationFailureCount + " place(s)\n")
            hashtagFailure += 1

        if len(initialArabizi[i].split(" ")) == len(initialGold[i].split(" ")):
            report.write("Hashtag Accuracy: " + hashtagAccuracy(initialArabizi[i], initialGold[i]) + "\n")

    if len(initialArabizi[i].split(" ")) != len(initialGold[i].split(" ")):
        report.write("ERROR: Incorrect alignment in intial output\n")
        incorrectInitialAlignment += 1
    else:
        initialAccuracy = accuracy(initialArabizi[i], initialGold[i])
        intInitialAccuracy = round(int(float(initialAccuracy)), -1)
        if intInitialAccuracy in initialAccuracyBin:
            initialAccuracyBin[intInitialAccuracy] += 1
        else:
            initialAccuracyBin[intInitialAccuracy] = 1
        report.write("Initial accuracy: " + initialAccuracy + "\n")

    if len(intermediateArabizi[i].split(" ")) != len(intermediateGold[i].split(" ")):
        incorrectIntermediateAlignment += 1
        report.write("ERROR: Incorrect alignment in intermediate output\n")
    else:
        intermediateAccuracy = accuracy(intermediateArabizi[i], intermediateGold[i])
        intIntermediateAccuracy = round(int(float(intermediateAccuracy)), -1)
        if intIntermediateAccuracy in intermediateAccuracyBin:
            intermediateAccuracyBin[intIntermediateAccuracy] += 1
        else:
            intermediateAccuracyBin[intIntermediateAccuracy] = 1
        report.write("Source aligned final accuracy: " + intermediateAccuracy + "\n")

        incorrectPlusToken += calcIncorrectPlusToken(intermediateArabizi[i], intermediateGold[i])
        missingPlusToken += calcMissingPlusToken(intermediateArabizi[i], intermediateGold[i])
        incorrectMinusToken += calcIncorrectMinusToken(intermediateArabizi[i], intermediateGold[i])
        missingMinusToken += calcMissingMinusToken(intermediateArabizi[i], intermediateGold[i])

    if len(finalArabizi[i].split(" ")) != len(finalGold[i].split(" ")):
        incorrectFinalAlignment += 1
        report.write("ERROR: Incorrect alignment in final output\n")

    report.write("\n")

report.close()


print("\tInitial Accuracy Bins".expandtabs(35))
prettyBins(initialAccuracyBin)
print("\tSource Aligned Final Accuracy Bins".expandtabs(35))
prettyBins(initialAccuracyBin)
print()
lineBreakdown(initialArabizi, initialGold, intermediateArabizi, intermediateGold, 1000)
print()
print("Overall Initial Accuracy: " + overallAccuracy(initialArabizi, initialGold))
print("Overall Source Aligned Final Accuracy: " + overallAccuracy(intermediateArabizi, intermediateGold))
print()
print("Total incorrect hashtag errors: " + str(incorrectHashtag))
print("Total missing hashtags: " + str(hashtagFailure))
print("Overall Hashtag Accuracy: " + str(round((globalHashtagCorrect/globalHashtagTotal)*100, 1)))
print()
print("Incorrect [+] tokens: {}; Incorrect [-] tokens: {}; Total: {}".format(str(incorrectPlusToken), str(incorrectMinusToken), str(incorrectPlusToken + incorrectMinusToken)))
print("Missing [+] tokens: {}; Missing [-] tokens: {}; Total: {}".format(str(missingPlusToken), str(missingMinusToken), str(missingPlusToken + missingMinusToken)))
print()
print("Total incorrect initial alignment errors: " + str(incorrectInitialAlignment))
print("Total incorrect intermediate alignment errors: " + str(incorrectIntermediateAlignment))
print("Total incorrect final alignment errors: " + str(incorrectFinalAlignment))
print("{}\n".format("-" * 97))