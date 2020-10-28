"""
Usage: python3 preprocess_fasttext_data.py --input_file --output_file --writing_system=latin --copy_marker=#
"""

import sys
import unicodedata as ud
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', action="store", dest='input_file', default="splits_ldc/source/source-without-dev-test.arabizi")
parser.add_argument('--output_file', action="store", dest='output_file', default="splits_ldc/source/source-without-dev-test-preprocessed.arabizi")
parser.add_argument('--writing_system', action="store", dest='writing_system', default="latin")
parser.add_argument('--copy_marker', action="store", dest='copy_marker', default="#")
args = parser.parse_args()

def allNonAscii(word):
    for char in word:
        if ord(char) < 128:
            return False
    
    return True

def copyNonAscii(input_line, copy_marker):
    input_words = input_line.split()
    
    for i in range(len(input_words)):
        if allNonAscii(input_words[i]):
            input_words[i] = copy_marker

    return " ".join(input_words)

def isPunctuation(word):
    for char in word:
        if char not in ".,?!'\":;-()[]}{":
            return False

    return True

def copyTextEmojiAndPunctuation(input_line, copy_marker):

    input_words = input_line.split()
    
    for i in range(len(input_words)):
        if isPunctuation(input_words[i]):
            input_words[i] = copy_marker

        else:
            # Handling <3 separately first
            match = re.search(r'(<3)+', input_words[i], re.IGNORECASE)
            if match and match.group(0) == input_words[i]:
                input_words[i] = copy_marker

            match = re.search(r'[=:;8xX>^()$*@][-_.\'"]*[XxOoPpsSDVv)(\][/\\Cc3><}{@|:;=0*L$^~]+', input_words[i], re.IGNORECASE) 
            if match:
                emoticon = match.group(0)
                if emoticon == input_words[i]:
                    input_words[i] = copy_marker

    return " ".join(input_words)

def removeAccents(line):
    
    newLine = []
    for word in line.split(" "):
        nfkd_form = ud.normalize('NFKD', word)
        res = "".join([c for c in nfkd_form if not ud.combining(c)])
        newLine.append(res.replace(' ', ''))
    return " ".join(newLine)

def compress(line, limit):

    ans = ""
    currChar = ""
    currCharCounter = 1

    compressThese = '23567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ '

    for i in line:
        if i == currChar:
            currCharCounter += 1
        else:
            currChar = i
            currCharCounter = 1
            
        if currCharCounter < limit + 1 or i not in compressThese:
            ans += i
    
    return ans

def preprocess(line):
    line = line.strip()
    line = compress(line, 2)
    line = line.lower()
    line = removeAccents(line)
    if args.writing_system == "latin":
        line = copyNonAscii(line, args.copy_marker)
    line = copyTextEmojiAndPunctuation(line, args.copy_marker)
    return line

rawFile = open(args.input_file, "r")
newFile = open(args.output_file, "w")

count = 1
for line in rawFile:
    if count % 10000 == 0:
        print("Lines processed:", count)

    newLine = preprocess(line)
    newFile.write(newLine + "\n")
    
    count += 1

rawFile.close()
newFile.close()