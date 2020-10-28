"""
Usage: python3 getSourceAndTarget.py [split/folder-of-xml-files] [split/source.arabizi] [split/word-aligned-target.gold] [split/sentence-aligned-target.gold]
"""

import xml.etree.ElementTree as ET
import os
import sys
import re
import unicodedata as ud

def fixPunctuation(line):
    punctuationDict = {"؛": ";", "؟": "?", "،": ","}
    newLine = []
    for word in range(len(line)):
        newWord = ""
        for char in range(len(line[word])):
            currCharacter = line[word][char]
            if currCharacter in punctuationDict:
                newWord += punctuationDict[currCharacter]
            else:
                newWord += currCharacter
        newLine.append(newWord)
    return newLine

def removeNones(arr):
    return list(filter(lambda x: x.text != None, arr))

def checkForForeignChar(word):
    for char in word:
        if ord(char) > 127:
            return True
    
    return False

def wordMadeOfSameChar(word):
    firstChar = word[0]
    for char in word:
        if char != firstChar:
            return False
    return True

def onlyHashtags(word):
    for char in word:
        if char != "#":
            return False
    return True

def countForeignChars(word):
    count = 0
    for char in word:
        if ord(char) > 127:
            count += 1

    return count


def fixHash(goldWord, annotatedArabiziWord):
    if len(goldWord) == len(annotatedArabiziWord) or goldWord == "#":
        return annotatedArabiziWord
    
    if onlyHashtags(goldWord):
        return annotatedArabiziWord

    if goldWord.count("#") == 1:
        
        if len(annotatedArabiziWord) == 1:
            return goldWord.replace("#", annotatedArabiziWord)

        # Handling <3 and <3<3 separately
        if annotatedArabiziWord.find("<3<3") != -1:
            return goldWord.replace("#", "<3<3")

        if annotatedArabiziWord.find("<3") != -1:
            return goldWord.replace("#", "<3")

        # Handling (y) separately
        if annotatedArabiziWord.find("(y)") != -1:
            return goldWord.replace("#", "(y)")
    
    # Other emojis
        match = re.search(r'[=:;8xX>^()$*@-][-_.\'"]*[XxOoPpsSDVvFfEe&)(\][/\\Cc3><}{@|:;=0*L$^~-]+', annotatedArabiziWord, re.IGNORECASE) 
        # Text Emojis
        if match:
            return goldWord.replace("#", match.group(0))
        else:
        # Handle graphic emojies
            if checkForForeignChar(annotatedArabiziWord) and goldWord[-1]:
                replacement = ""
                for i in range(len(annotatedArabiziWord)-1, -1, -1):
                    if ord(annotatedArabiziWord[i]) < 128:
                        break
                    replacement += annotatedArabiziWord[i]
                return goldWord.replace("#", replacement)

            elif checkForForeignChar(annotatedArabiziWord) and goldWord[0]:
                replacement = ""
                for i in range(len(annotatedArabiziWord)):
                    if ord(annotatedArabiziWord[i]) < 128:
                        break
                    replacement += annotatedArabiziWord[i]
                return goldWord.replace("#", replacement)
            
        # Handling characters such as @$* etc at start or end of word
            elif goldWord[-1] == "#":
                replacement = ""
                for i in range(len(annotatedArabiziWord)-1, -1, -1):
                    if annotatedArabiziWord[i] not in "~!@#$%^&*()-_=+\\|]}[{\":;?/>.<,'":
                        break
                    replacement += annotatedArabiziWord[i]
                return goldWord.replace("#", replacement)

            elif goldWord[0] == "#":
                replacement = ""
                for i in range(len(annotatedArabiziWord)):
                    if annotatedArabiziWord[i] not in "~!@#$%^&*()-_=+\\|]}[{\":;?/>.<,'":
                        break
                    replacement += annotatedArabiziWord[i]
                return goldWord.replace("#", replacement)

            else:
                indexOfHash = goldWord.index("#")
                return goldWord.replace("#", annotatedArabiziWord[indexOfHash])
            
    # In case of brackets
    elif goldWord.count("#") == 2 and annotatedArabiziWord.count("(") == 1 and annotatedArabiziWord.count(")") == 1:
        goldWord = goldWord.replace("#", "(", 1)
        goldWord = goldWord.replace("#", ")")
        return goldWord
        
    else:
        if wordMadeOfSameChar(annotatedArabiziWord):
            return goldWord.replace("#", annotatedArabiziWord[0])
        
        elif len(annotatedArabiziWord) == 1:
            return annotatedArabiziWord
        
        elif goldWord.count("#") == annotatedArabiziWord.count("<3"):
            return goldWord.replace("#", "<3")
        
        elif goldWord.count("#") == countForeignChars(annotatedArabiziWord):
            for char in annotatedArabiziWord:
                if ord(char) > 127:
                    goldWord = goldWord.replace("#", char, 1)
            return goldWord

        else:
            for i in range(goldWord.count("#")):
                match = re.search(r'[=:;8xX>^()$*<@-][-_.\'"]*[XxOoPpsSDVvFfEe&)(\][/\\Cc3><}{@0*L$^~|3]', annotatedArabiziWord, re.IGNORECASE) 
                if match:
                    goldWord = goldWord.replace("#", match.group(0), 1)
                    annotatedArabiziWord = annotatedArabiziWord.replace(match.group(0), "", 1)
                elif len(annotatedArabiziWord) == 1:
                    goldWord = goldWord.replace("#", annotatedArabiziWord)

            return goldWord

def isWholeWordForeign(word):
    for char in word:
        if ord(char) < 128:
            return False
    return True

def removeSeparationTokens(line, file):
    line = line.replace("[+] ", "")
    line = line.replace("[-]", " ")
    line = line.replace("[+]", "")
    return line

dirs = os.listdir(sys.argv[1])
dirs.sort()

arabiziOutput = open(sys.argv[2], "w") # This will be the input for seq2seq
intermediateOutput = open(sys.argv[3], "w") # We'll use this as our seq2seq model gold
finalOutput = open(sys.argv[4], "w") # This is the final target without any [-] and [+]

for i in dirs:
    if i[-3:] != "xml":
        continue

    tree = ET.parse(sys.argv[1] + i)
    root = tree.getroot()

    tracker = 0
    allWords = root.findall("./su/annotated_arabizi/token")
    allRawGold = removeNones(root.findall("./su/corrected_transliteration"))
    totalLinesInFile = len(allRawGold)
    lineCounter = 0
    for source in root.findall("./su/source"):
        if lineCounter >= totalLinesInFile:
            continue
            
        numLines = 0
        if source.text == None:
            continue

        currLineSourceWords = source.text.strip().split()
        arabiziOutput.write(source.text.strip() + "\n")

        currGoldLine = allRawGold[lineCounter].text.strip().split()
        currGoldLine = fixPunctuation(currGoldLine)

        numLines = len(source.text.split())

        if tracker >= len(allWords):
            continue

        wordsOfThisLineCounter = 0
        for j in range(tracker, tracker + numLines):

            if "[+]" in currGoldLine[wordsOfThisLineCounter][-3:] and j + 1 < tracker + numLines and "tag" in allWords[j+1].attrib and allWords[j+1].attrib["tag"] == "foreign":
                currGoldLine[wordsOfThisLineCounter] = currGoldLine[wordsOfThisLineCounter][:-3]

            if ("tag" in allWords[j].attrib and allWords[j].attrib["tag"] in ["punctuation", "foreign"]) or isWholeWordForeign(allWords[j].text):
                currGoldLine[wordsOfThisLineCounter] = currLineSourceWords[wordsOfThisLineCounter]

            if "#" in currGoldLine[wordsOfThisLineCounter]:
                currGoldLine[wordsOfThisLineCounter] = fixHash(currGoldLine[wordsOfThisLineCounter], allWords[j].text)
            wordsOfThisLineCounter += 1
        
        goldLineWithSeparationTokens = " ".join(currGoldLine)
        intermediateOutput.write(goldLineWithSeparationTokens + "\n")

        if len(source.text.strip().split()) != len(currGoldLine):
            pass
            
        goldLineWithoutSeparationTokens = removeSeparationTokens(goldLineWithSeparationTokens, i)
        finalOutput.write(goldLineWithoutSeparationTokens + "\n")
        tracker += numLines
        lineCounter += 1
        
arabiziOutput.close()
intermediateOutput.close()
finalOutput.close()