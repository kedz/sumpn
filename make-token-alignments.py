from __future__ import division
import sys
import os
from collections import defaultdict
import math
import random
import re
import json
from itertools import product
from data_utils import read_document, replace_entities, get_document_paths
from data_utils import stopwords, unk_id, sw_id
import yaml
from multiprocessing import Pool

def init_doc_meta(doc, max_sent):

    id2token = list()
    id2sent = list()
    sent2token_ids = list()
    sent2tokens = list()
    token_sets = list()

    for s, sentence in enumerate(doc["sentences"][:max_sent]):
        
        tokens = replace_entities(sentence["tokens"], doc["entities"])
        token_ids = [id for id, token in enumerate(tokens, len(id2token))]
        sent2token_ids.append(token_ids)
        sent2tokens.append(tokens)
        id2token.extend(tokens)
        id2sent.extend([s] * len(token_ids))

        token_sets.append(
            set([token for token in tokens if token not in stopwords]))

    return id2token, id2sent, sent2tokens, sent2token_ids, token_sets

def find_sentence_support(highlight_token_set, doc_token_sets):

    sents = list()
    max_score = 1

    while max_score > 0:

        max_score = 0
        max_sent = None
        for dts, doc_token_set in enumerate(doc_token_sets):
            score = len(doc_token_set.intersection(highlight_token_set))
            if score > max_score:
                max_score = score
                max_sent = dts
        if max_sent is not None:
            sents.append(max_sent)
            highlight_token_set -= doc_token_sets[max_sent]

    sents.sort()
    return sents

def find_token_alignments(source_tokens, source_token_ids, highlight_tokens, quotes):

    token2pos = defaultdict(list)

    for t, token in enumerate(source_tokens):
        if token not in stopwords:
            token2pos[token].append(source_token_ids[t])

    A = defaultdict(dict)
    for i, token in enumerate(highlight_tokens):
        if i in quotes:
            A[(1,i)][tuple([quotes[i]])] = 0   
        else:
            occurences = token2pos.get(
                token, [sw_id] if token in stopwords else [unk_id])

            for j in occurences:
                A[(1,i)][tuple([j])] = 0
    
    for l in xrange(2, len(highlight_tokens) + 1): #-- Length of span
        for s in xrange(len(highlight_tokens)-l+1): # -- Start of span

            for p in xrange(1,l): # -- Partition of span
                len_left = p
                len_right = l - p
                max_score = -9999999
                for seq_left, score_left in A[(len_left,s)].items():

                    for seq_right, score_right in A[(len_right,s+p)].items():
                        new_score = score_left + score_right
                        if seq_left[-1] >= 0 and seq_right[0] >= 0 and \
                            seq_left[-1] + 1 == seq_right[0]:
                                new_score += 1
                        

                        A[(l,s)][seq_left + seq_right] = new_score
                        if max_score < new_score: max_score = new_score
                for k,v in A[(l,s)].items():
                    if v < max_score: del A[(l,s)][k]

    return A[(len(highlight_tokens), 0)].keys()

def rescore_alignments(raw_token_alignments, id2sent):

    raw_alignments_scores = list()
    for raw_alignment in raw_token_alignments:
        uniq_sents = set()
        for a in raw_alignment: 
            if a >= 0:
                uniq_sents.add(id2sent[a])
        raw_alignments_scores.append((raw_alignment, len(uniq_sents)))

    raw_alignments_scores.sort(key=lambda x: x[1])
    
    alignments = list(raw_alignments_scores[0][0])  

    return alignments

def fill_stopword_alignments(source_tokens, source_token_ids, highlight_tokens,
            token_alignments):

    if len(source_tokens) > 0:
        
        last = len(token_alignments) - 1 
        
        for i in xrange(len(token_alignments) - 1, -1, -1):
            if token_alignments[i] != sw_id and token_alignments[i] != unk_id:
                last = i
                break
        
        for i in xrange(last - 1, -1, -1):
            if token_alignments[i] == sw_id:
                token = highlight_tokens[i]
                source_last = source_token_ids.index(token_alignments[last])
                try:
                    index = source_tokens[:source_last][::-1].index(token)
                    if index < 10:
                        token_alignments[i] = source_token_ids[
                            source_last - index - 1]
                except ValueError, e:
                    pass
            elif token_alignments[i] != unk_id: 
                last = i


def find_quotes(highlight_tokens, sent2tokens, sent2token_ids):

    support = list()
    hl_string = " ".join(highlight_tokens)
    for s, tokens in enumerate(sent2tokens):
        tstr = " ".join(tokens) 
        index = hl_string.find(tstr)
        if index > -1:
            #print "!!!!!!!!!!!!!!!!!!!!!"
            #print hl_string[:index]
            #print hl_string[index:index + len(tstr)]
            #print hl_string[index + len(tstr):]
            #print sent2token_ids[s]
            hl_string = "{} {} {}".format(
                hl_string[:index],
                " ".join("__{}__".format(i) for i in sent2token_ids[s]),
                hl_string[index + len(tstr):])
            support.append(s)

    quoted_tokens = hl_string.strip().split()
    if len(quoted_tokens) != len(highlight_tokens):
        raise Exception(
            "hl_string has wrong number of tokens!: {}".format(hl_string))

    quotes = dict()
    for i, token in enumerate(quoted_tokens):
        m = re.search(r"__(\d+)__", token)
        if m:
            quotes[i] = int(m.group(1))
    return quotes, support


def find_highlight_alignments(highlight, doc, meta):
    
    id2token, id2sent, sent2tokens, sent2token_ids, doc_token_sets = meta

    highlight_tokens = replace_entities(highlight["tokens"], doc["entities"])
    highlight_token_set = set(highlight_tokens)
    #print len(highlight_tokens)


    #print "###\n"
    quotes, quote_support = find_quotes(highlight_tokens, sent2tokens, sent2token_ids)
    for s in quote_support:
        highlight_token_set -= doc_token_sets[s]
    support = find_sentence_support(highlight_token_set, doc_token_sets)
    #print quote_support, support
   
    support.extend(quote_support) 
    support.sort()

    #for s in support:
    #    print " ".join(sent2tokens[s])

    #print
    #print " ".join(highlight_tokens)

    src_tokens = [token for sent in support for token in sent2tokens[sent]] 
    src_ids = [id for sent in support for id in sent2token_ids[sent]]

    raw_token_aligments = find_token_alignments(
            src_tokens, src_ids, highlight_tokens, quotes)
    token_alignments = rescore_alignments(raw_token_aligments, id2sent)

    fill_stopword_alignments(src_tokens, src_ids, highlight_tokens,
            token_alignments)
    
    sent_counts = defaultdict(int)
    for a in token_alignments:
        if a >= 0: sent_counts[id2sent[a]] += 1

    sent_counts = sent_counts.items()

    #print token_alignments

    if len(sent_counts) > 0:

        # Sourt counts first by sentence id, then by count. Since sort is 
        # stable count ties will be broken by earliest occurring support 
        # sentence.
        sent_counts.sort(key=lambda x: x[0])
        sent_counts.sort(key=lambda x: x[1], reverse=True)
        
        backbone = sent_counts[0][0]
        support = [sc[0] for sc in sent_counts[1:]]
        return backbone, support, token_alignments
            
    else:
        return None, list(), token_alignments

def process_document(args): 
    document_path, output_dir, max_input, max_highlight, no_overwrite = args
    output_path = os.path.join(output_dir, os.path.split(document_path)[1])
    if no_overwrite is True and os.path.exists(output_path):
        return
    print document_path

    doc = read_document(document_path)

    meta = init_doc_meta(doc, max_input)
    #print document_path 
    data = list()
    for highlight in doc["highlights"][:max_highlight]:
        backbone, support, alignments = find_highlight_alignments(
            highlight, doc, meta)
        data.append([backbone, support, alignments])

    with open(output_path, "w") as f:
        f.write(yaml.dump(data))

def main():

    import argparse

    hlp = "Find token level alignments."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--split', required=True, help="Data split to use.",
        choices=["train", "dev", "test"])
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument('--samples', default=-1, type=int,
        help="Number of samples. -1 indicates all data.")
    parser.add_argument('--random', default=False, action="store_true",
        help="Shuffle data.")
    parser.add_argument('--output', required=True,
        help="Output directory to write summaries.")
    parser.add_argument('--seed', default=1986, type=int,
        help="Random seed if random is true.")
    parser.add_argument('--max-input-sent', required=True, type=int,
        help="Maximum number of sentences to read from input.")
    parser.add_argument('--max-highlight', required=True, type=int,
        help="Maximum number of highlights to read.")
    parser.add_argument('--no-overwrite', required=False, action="store_true")

    args = parser.parse_args()
     
    arg2split = {"test": "test", "train": "training", "dev": "validation"}
    split = arg2split[args.split]

    input_path = os.path.join(args.data_path, args.corpus, split)
    if not os.path.exists(input_path):
        raise Exception("Path to data ({:s}) does not exist.".format(
            input_path))
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    random.seed(args.seed)

    document_paths = get_document_paths(input_path, args.samples, args.random)
    document_paths.reverse()

    args = [(dp, args.output, args.max_input_sent, 
             args.max_highlight, args.no_overwrite) 
            for dp in document_paths]
    
    pool = Pool(processes=4)
    n_docs = len(document_paths)
    #for arg in args:
    #    process_document(arg)
    for i, d in enumerate(pool.imap_unordered(process_document, args), 1):
    #for i, d in enumerate(results, 1):
    #    if i < 14835:
    #        continue
        #print document_path
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, n_docs, 100 * i / n_docs))
        sys.stdout.flush()
    #    pass
        #process_document(document_path, args.output, 
        #    args.max_input_sent, args.max_highlight)
    print
 


if __name__ == "__main__":
    main()
