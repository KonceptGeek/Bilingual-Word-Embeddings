__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import numpy as np
import sys


class WordEmbedding(object):
    """
    WordEmbedding class holds the embeddings for each word. It loads the embeddings during init from the provided
    file path.
    """
    def __init__(self, filePath):
        self._wordEmbeddings, self._embeddingDimensions = self._readEmbeddings(filePath)

    def getEmbeddingForWord(self, word):
        """
        Get Embeddings for the provided word.
        :param word:
        :return:
        """
        if word in self._wordEmbeddings:
            return self._wordEmbeddings[word]
        return None

    def getAvgEmbeddingForPhrase(self, phrase):
        """
        Get the average embedding for provided phrase. Splits the phrase on space to get each word, for each word
        if we have an embedding, get it. Finally for all the embeddings collected, calculate element wise average
        to get average embedding for the phrase.
        :param phrase:
        :return:
        """
        words = [word.strip().lower() for word in phrase.split(' ')]
        vectors = []
        errorOccurred = False
        for word in words:
            if word in self._wordEmbeddings:
                vectors.append(self._wordEmbeddings[word])
        if len(vectors) == 0:
            sys.stderr.write('No Embedding Vector for phrase: {} ||| Using Zeros Vector.\n'.format(phrase))
            vectors.append(np.zeros(self._embeddingDimensions))
            errorOccurred = True
        vectors = np.array(vectors)
        avgVector = np.mean(vectors, axis=0)
        return avgVector, errorOccurred

    def _readEmbeddings(self, filePath):
        """
        Read word embeddings from the provided file path. It loads all the embeddings in memory.
        :param filePath:
        :return:
        """
        sys.stdout.write("Reading embeddings from: {}\n".format(filePath))
        result = {}
        embeddingDimensions = 0
        with open(filePath) as inFile:
            for line in inFile:
                lineSplit = line.split(' ')
                word = lineSplit[0].strip()
                embeddings = np.array([float(val.strip()) for val in lineSplit[1].strip().split(',')])
                result[unicode(word, "utf-8").encode("utf-8")] = embeddings
                embeddingDimensions = len(embeddings)
        sys.stdout.write("Finished reading {} embeddings from: {} of size {}\n".format(len(result), filePath, embeddingDimensions))
        return result, embeddingDimensions


if __name__ == '__main__':
    englishEmbeddings = WordEmbedding('/cs/natlang-user/jasneet/Thesis/Code/MT/Zou-et-al-2013/resources/En_vectors.txt')
    print englishEmbeddings.getEmbeddingForWord('male')
    print englishEmbeddings.getEmbeddingForWord('female')
    print englishEmbeddings.getAvgEmbeddingForPhrase('male female')