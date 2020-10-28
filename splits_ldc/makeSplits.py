"""
Usage: python3 makeSplits.py [path/to/all/xml/files] [split.txt] [splitFolder]
"""

import xml.etree.ElementTree as ET
import os
import sys
import shutil
from pathlib import Path

pathToXMLFiles = sys.argv[1]
# Read the txt file
splitTxt = open(sys.argv[2], "r")

folder = sys.argv[3]
Path(folder).mkdir(parents=True, exist_ok=True)

for xmlFile in splitTxt.readlines():
    xmlFile = xmlFile.strip()

    # Copy the xml to the correct split folder
    shutil.copy(pathToXMLFiles + xmlFile, folder)

splitTxt.close()

