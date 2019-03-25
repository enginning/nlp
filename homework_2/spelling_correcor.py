"""
    Ref: http://norvig.com/spell-correct.html
    edited by enginning for homework_2 on 20190325
"""

################ Spelling Corrector 

import re
from collections import Counter

def words(text): 
    return re.findall(r'\w+', text.lower())

with open('corpus.txt', encoding = 'utf-8') as f:  
    process = words(f.read())
    WORDS = Counter(process)
    num = len(process) /1000000
    print("the corpus has {:.2f} million words".format(num))
    print("the corpus's vacabulary is ", len(WORDS))
    
def P(word, N = sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    # print("candidate: ", candidates(word), " , type of candidate is " , type(candidates(word)))
    return max(candidates(word), key = P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    # print(set(w for w in words if w in WORDS))
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1)) 

################ Test Code 

def unit_tests():
    assert correction('speling') == 'spelling'              # insert
    assert correction('korrectud') == 'corrected'           # replace 2
    assert correction('bycycle') == 'bicycle'               # replace
    assert correction('inconvient') == 'inconvenient'       # insert 2
    assert correction('arrainged') == 'arranged'            # delete
    assert correction('peotry') =='poetry'                  # transpose
    assert correction('peotryy') =='poetry'                 # transpose + delete
    assert correction('word') == 'word'                     # known
    assert correction('quintessential') == 'quintessential' # unknown
    assert words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(words('This is a test. 123; A TEST this is.')) == (
           Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
    return 'unit_tests pass'

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    scale = len(tests)
    factor = scale / 100
    print("test".center(112, "-"))
    
    good, unknown, count= 0, 0, 0
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
        count += 1
        # print processing  bar
        if (count % round(factor)) == 0:
            scale_bar = int(scale / factor)
            a = '#' * int(count / factor)
            b = '.' * (scale_bar - int(count / factor))
            c = float(count / scale) * 100
            dur = time.clock() - start
            print("\r{:^3.0f}%[{}->{}]{:.2f}s, correct rate: {:.0%}".format(c, a, b, dur, good / count), end='')
    minutes = (int)(dur - start) // 60
    seconds = (int)(dur - start) % 60
    print("\nTotal test time: {0:} minutes {1:} seconds".format(minutes, seconds))
    print('{:.1f}% of {} correct ({:.1f}% unknown) at {:.0f} words per second\n'
          .format(100 * good / scale, scale, 100 * unknown / scale, scale / dur)) 

def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

if __name__ == '__main__':
    print(unit_tests())
    spelltest(Testset(open('testset_1.txt')))
    spelltest(Testset(open('testset_2.txt')))