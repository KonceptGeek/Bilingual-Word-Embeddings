__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

"""
Create new phrase_table file
Read phrase table line by line
for each line
    calculate bilingual phrase similarity
    augment the phrase line
    write new line to the temporary phrase_table
"""

from wordEmbedding import WordEmbedding
from scipy import linalg, dot
import numpy as np
import HTMLParser
import gzip
import sys
import os

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

hparser = HTMLParser.HTMLParser()


def readPhraseTable(filePath):
    with gzip.open(filePath, 'r') as inFile:
        for line in inFile:
            yield line.strip('\n')


def getCleanedPhrase(phrase):
    """
    Clean the phrase by converting each word to utf-8 format and separating each word by 1 space
    :param phrase:
    :return:
    """
    phraseSplit = phrase.strip().split(' ')
    cleanedPhrase = ''
    for word in phraseSplit:
        try:
            parsedWord = hparser.unescape(word)
        except UnicodeDecodeError:
            parsedWord = word

        if type(parsedWord) != unicode:
            parsedWord = unicode(parsedWord, "utf-8").encode("utf-8")
        else:
            parsedWord = parsedWord.encode("utf-8")
        cleanedPhrase += parsedWord
        cleanedPhrase += ' '
    return cleanedPhrase.strip()


def calculateCosineDistance(sourceEmbedding, targetEmbedding):
    """
    Calculate cosine distance between sourceEmbedding and targetEmbedding.
    cos(A, B) = (A.B)/(||A|| ||B||)
    :param sourceEmbedding:
    :param targetEmbedding:
    :return:
    """
    if np.count_nonzero(sourceEmbedding) > 0 and np.count_nonzero(targetEmbedding) > 0:
        cosineSimilarity = dot(sourceEmbedding, targetEmbedding)/(linalg.norm(sourceEmbedding) *
                                                                  linalg.norm(targetEmbedding))
    else:
        cosineSimilarity = 0.0
    cosineDistance = 1.0 - cosineSimilarity
    return cosineDistance


def augmentPhraseTable(inputPhraseTablePath, sourceEmbeddingsPath, targetEmbeddingsPath, outputPhraseTablePath):
    """
    Augment Phrase Table - Read input phrase table. For each bilingual phrase pair, create average embeddings vector
    for both source and target. Calculate the cosine distance between the vectors to form a semantic similarity feature
    for the decoder.

    :param inputPhraseTablePath:
    :param sourceEmbeddingsPath:
    :param targetEmbeddingsPath:
    :param outputPhraseTablePath:
    :return:
    """
    phraseTableData = readPhraseTable(inputPhraseTablePath)
    targetEmbeddings = WordEmbedding(targetEmbeddingsPath)
    sourceEmbeddings = WordEmbedding(sourceEmbeddingsPath)

    sys.stdout.write("\nAUGMENTING PHRASE TABLE: {}\n\n".format(inputPhraseTablePath))

    with gzip.open(outputPhraseTablePath, 'wb') as outFile:
        for i, line in enumerate(phraseTableData):
            lineSplit = line.split('|||')
            sourcePhrase = lineSplit[0]
            sourcePhrase = getCleanedPhrase(sourcePhrase)
            avgSourceEmbedding, errorOccurred = sourceEmbeddings.getAvgEmbeddingForPhrase(sourcePhrase)
            if errorOccurred:
                sys.stderr.write('SOURCE PHRASE: ' + line + '\n')

            targetPhrase = lineSplit[1]
            targetPhrase = getCleanedPhrase(targetPhrase)
            avgTargetEmbedding, errorOccurred = targetEmbeddings.getAvgEmbeddingForPhrase(targetPhrase)
            if errorOccurred:
                sys.stderr.write('TARGET PHRASE: ' + line + '\n')

            cosineDistance = calculateCosineDistance(avgSourceEmbedding, avgTargetEmbedding)
            lineSplit[2] = lineSplit[2].strip() + ' ' + str(cosineDistance)

            updatedLine = ''
            for val in lineSplit:
                updatedLine += val.strip()
                if len(val.strip()) > 0:
                    updatedLine += ' ||| '
                else:
                    updatedLine += '||| '

            updatedLine = updatedLine.strip(' |||')

            outFile.write(updatedLine+'\n')

            if i % 10000 == 0:
                sys.stdout.write(str(i)+'\n')

    sys.stdout.write("\nAUGMENTING PHRASE TABLE COMPLETED: {}\n\n".format(outputPhraseTablePath))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phrase-table", help="The input phrase table")
    parser.add_argument("-s", "--source-embeddings", help="The source language word embeddings")
    parser.add_argument("-t", "--target-embeddings", help="The target language word embeddings")
    parser.add_argument("-o", "--output-phrase-table", help="The output phrase table")
    args = parser.parse_args()
    if not (args.phrase_table or args.source_embeddings or args.target_embeddings or args.output_phrase_table):
        parser.error('Missing arguments')

    augmentPhraseTable(args.phrase_table, args.source_embeddings, args.target_embeddings, args.output_phrase_table)
