from __future__ import division
import sys
import os
import argparse
import yaml
from data_utils2 import read_line_document

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]: 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C

def backTrack(C, X, Y, i, j):
    if i == 0 or j == 0:
        return []
    elif X[i-1] == Y[j-1]:
        return backTrack(C, X, Y, i-1, j-1) + [j-1]  #+ X[i-1]
    else:
        if C[i][j-1] > C[i-1][j]:
            return backTrack(C, X, Y, i, j-1)
        else:
            return backTrack(C, X, Y, i-1, j) + [-1]

def backTrackAll(C, X, Y, i, j):
    if i == 0 or j == 0:
        return set(tuple())
    elif X[i-1] == Y[j-1]:
        return set([Z + tuple([j-1]) for Z in backTrackAll(C, X, Y, i-1, j-1)])
    else:
        R = set()
        if C[i][j-1] >= C[i-1][j]:
            R.update(backTrackAll(C, X, Y, i, j-1))
        if C[i-1][j] >= C[i][j-1]:
            R.update(backTrackAll(C, X, Y, i-1, j))
        return R


def find_lcs(target, sequences):
    seq2lcs = list()
    for sequence in sequences:
        C = LCS(target, sequence)
        align = backTrack(C, target, sequence, len(target), len(sequence))
        seq2lcs.append(align)

    I = [i for i in range(len(seq2lcs))]
    matches = [len([a for a in seq2lcs[i] if a > -1]) for i in I]
    backbone = max(I, key=lambda x: matches[x])
    lcs = seq2lcs[backbone]
    if len(lcs) < len(target):
        lcs = [-1] * (len(target) - len(lcs)) + lcs

    return backbone, lcs, matches[backbone]

def process(doc):

    all_tokens = [tok.lower() for s in doc.sentences
                  for tok in s]
    bad_verbs = set([
        "am", "is", "are", "was", "were", "be", "been", "being", 
        "will", "would", "have", "has", "had", "having", "made", 
        "mr.", "mr", "mrs.", "mrs", "ms", "ms.", "dr", "dr."])
    bad_pos = set(["DT", "IN", "PRP", "PRP$", "POS", ",", "''", "CC", "TO", "RB", "MD", "WRB", "WDT"])

    sent2tokens = list()
    id2sent = list()
    for s, sentence in enumerate(doc.sentences):
        tokens = list()
        for token in sentence:
            id2sent.append(s)
            tokens.append(token.lower())
        sent2tokens.append(tokens)

    results = list()
    for highlight in doc.highlights:
        #print " ".join([h.token + "/" + h.pos for h in highlight])
        
        highlight_tokens = [t.lower() for t in highlight]
        highlight_stopped = list()
        for token in highlight:
            form = token.lower()
            if token.pos not in bad_pos and form not in bad_verbs:
                highlight_stopped.append(form)
            else:
                highlight_stopped.append("__SW__")

        matches = 1

        supports = list()
        relative_alignments = list()
        while matches > 0:
        
            support, alignment, matches = find_lcs(
                highlight_stopped, sent2tokens)
            highlight_stopped = [highlight_stopped[i] if a == -1 else "__SW__"
                                 for i, a in enumerate(alignment)] 
            if matches > 0:
                supports.append(support)
                relative_alignments.append(alignment)
        #print "HIGHLIGHT"
        #print " ".join(highlight_tokens)
        #print "SUPPORTS"
        alignments = [-1] * len(highlight_tokens)
        for support, relative_alignment in zip(supports, relative_alignments):
            #print " ".join(sent2tokens[support]) 
            offset = sum([len(sent) for sent in sent2tokens[:support]])
            for i, a in enumerate(relative_alignment):
                if a != -1:
                   alignments[i] = a + offset
                   assert(all_tokens[a + offset] == highlight_tokens[i])
        uniq_supports = list()
        used = set()
        for s in supports:
            if s not in used:
                uniq_supports.append(s)
                used.add(s)

        if len(uniq_supports) > 0:
            backbone = uniq_supports[0]
            support = uniq_supports[1:]
            results.append((backbone, support, alignments))
        else:
            results.append((None, list(), alignments))
    return results


def main():

    hlp = "Greedy aligner using longest common subsequence algo."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--input', required=True,
        help="Location of processed content tsv file.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
 
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.input, "r") as f:
        input_lines = [line for line in f]
    n_data = len(input_lines)

    def yaml_iter():
        for i, line in enumerate(input_lines, 1):
            sys.stdout.write("Aligned {:8.4f}% \r".format(100 * i / n_data))
            sys.stdout.flush()
            doc = read_line_document(line)
            res = process(doc)
            yield (doc.filename, res)


    with open(args.output, "w") as f:
        yaml.dump_all(yaml_iter(), f, explicit_start=True)


if __name__ == "__main__":
    main()
