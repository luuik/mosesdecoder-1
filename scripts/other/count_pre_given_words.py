#!/usr/bin/python

# input: one file = tokenised corpus
# output: one file = ngram counts
# * unigram = count of prefixes
# * bigram = count of prefixes given previous word
# * trigram = count of prefixes given two previous words

import sys
import re

with open(sys.argv[1], 'r') as inputfile:
    for line in inputfile:
        line = line.rstrip()
        tok = re.split(' ',line)
        for i in range(0, len(tok)):
            for j in range(0, len(tok[i])):
                prefix = tok[i][0:j+1]
                print prefix+" 1"
                if (i>0):
                    bigram = tok[i-1]+" "+prefix
                    print bigram+" 1"
                    if (i>1):
                        trigram = tok[i-2]+" "+bigram
                        print trigram+" 1"
