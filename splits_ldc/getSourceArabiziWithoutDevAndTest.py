"""
Usage: python3 getSourceArabiziWithoutDevAndTest.py [path/to/source-xml-files] [path/to/dev-xml-files] [path/to/test-xml-files] [newSource.arabizi]
"""

import xml.etree.ElementTree as ET
import sys
import os
from pathlib import Path

def excludeLines(path, sf, sourcePath):
    global lineCount
    
    dtTree = ET.parse(path + "/" + sf[:-9] + ".transli.xml")
    dtRoot = dtTree.getroot()
    dtAllMessages = dtRoot.findall("./su/messages/message")

    sourceTree = ET.parse(sourcePath + "/" + sf)
    sourceRoot = sourceTree.getroot()
    sourceAllMessages = sourceRoot.findall("./messages/message")

    dtMessageIds = []
    for dtm in dtAllMessages: dtMessageIds.append(dtm.attrib["id"]) 
    for sm in sourceAllMessages:
        if sm.attrib["id"] not in dtMessageIds and sm[0].text != None:
            newSource.write(sm[0].text + "\n")
            lineCount += 1


sourceFiles = os.listdir(sys.argv[1])
sourceFiles.sort()
devFiles = os.listdir(sys.argv[2])
devFiles.sort()
testFiles = os.listdir(sys.argv[3])
testFiles.sort()

directory = os.path.dirname(sys.argv[4])
Path(directory).mkdir(parents=True, exist_ok=True)
newSource = open(sys.argv[4], "w")
fileCount = 1
lineCount = 0
for sf in sourceFiles:

    if fileCount % 200 == 0:
        print(str(fileCount) + " files read. " + str(lineCount) + " lines processed.")

    if sf[-3:] != "xml":
        continue

    if sf[:-9] + ".transli.xml" in devFiles:
        excludeLines(sys.argv[2], sf, sys.argv[1])
    
    elif sf[:-9] + ".transli.xml" in testFiles:
        excludeLines(sys.argv[3], sf, sys.argv[1])

    else:
        sourceTree = ET.parse(sys.argv[1] + "/" + sf)
        sourceRoot = sourceTree.getroot()
        sourceAllMessages = sourceRoot.findall("./messages/message/body")
        for message in sourceAllMessages:
            if message.text != None:
                newSource.write(message.text + "\n")
                lineCount += 1

    fileCount += 1

newSource.close()