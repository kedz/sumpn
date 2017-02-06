from __future__ import division
import sys
import os
from data_utils import read_document, read_vocab, get_document_paths
from data_utils import unk_id, sw_id, preprocess_tokens
from itertools import izip
import yaml
import numpy as np

def collect_split_stats(data_dir, alignments_dir, vocab_out):

    document_paths = get_document_paths(data_dir)
    alignments_paths = get_document_paths(alignments_dir)

    backbone_counts = list()
    highlight_counts = list()
    support_counts = list()
    aligned_counts = list()
    unaligned_ent_counts = list()
    unaligned_counts = list()
    unaligned_common_counts = list()

    for doc_path, align_path in izip(document_paths, alignments_paths):
        
        if not os.path.basename(doc_path) == os.path.basename(align_path):
            raise Exception(
                 "Alignments directory does not contain one file for every " \
                 "file in data path.")
        
        doc = read_document(doc_path)

        with open(align_path, "r") as f:
            alignments = yaml.load(f)
        
        backbone_count = 0
        for a in xrange(len(alignments)):
            backbone, support, token_alignments = alignments[a]
            if backbone is not None:
                backbone_count += 1
            support_counts.append(len(support))

            highlight_tokens = doc["highlights"][a]["tokens"]
            pp_highlight_tokens = preprocess_tokens(
                    highlight_tokens, doc["entities"]) 

            aligned_tokens = list()
            unaligned_tokens = list()
            unaligned_common_tokens = list()
            unaligned_entity_tokens = list()

            for token, align in izip(pp_highlight_tokens, token_alignments):
                if align == unk_id or align == sw_id: 
                    unaligned_tokens.append(token)
                    if token in vocab_out:
                        unaligned_common_tokens.append(token)
                    elif token == "__ENTITY__":
                        unaligned_entity_tokens.append(token)
                else:
                    aligned_tokens.append(token)
                
            unaligned_ent_counts.append(len(unaligned_entity_tokens))

            aligned_counts.append(len(aligned_tokens))
            unaligned_counts.append(len(unaligned_tokens))
            unaligned_common_counts.append(len(unaligned_common_tokens))

        backbone_counts.append(backbone_count)
        highlight_counts.append(len(alignments))

    print "% highlights w/o alignments", \
        1 - np.sum(backbone_counts) / np.sum(highlight_counts)
    print "macro avg. support", np.mean(support_counts)

    aligned_counts = np.array(aligned_counts)
    unaligned_counts = np.array(unaligned_counts)
    unaligned_common_counts = np.array(unaligned_common_counts)
    total_tokens = aligned_counts + unaligned_counts
    
    macro_avg_align_recall = (aligned_counts / total_tokens).mean()
    micro_avg_align_recall = aligned_counts.sum() / total_tokens.sum()
    
    macro_avg_unalign_recall = (unaligned_common_counts / total_tokens).mean()
    micro_avg_unalign_recall = \
        unaligned_common_counts.sum() / total_tokens.sum()

    macro_avg_unalign_ent_recall = (unaligned_ent_counts / total_tokens).mean()

    macro_avg_max_recall = \
        ((unaligned_common_counts + aligned_counts) / total_tokens).mean()
    micro_avg_max_recall = \
        (unaligned_common_counts.sum() + aligned_counts.sum()) \
        / total_tokens.sum()

    print "avg. token count", total_tokens.mean()
    print "macro avg. align. recall", macro_avg_align_recall
    print "micro avg. align. recall", micro_avg_align_recall
    print "macro avg. unalign. recall", macro_avg_unalign_recall
    print "micro avg. unalign. recall", micro_avg_unalign_recall
    print "macro avg. unalign. ent recall", macro_avg_unalign_ent_recall

    print "macro avg. max recall", macro_avg_max_recall
    print "micro avg. max recall", micro_avg_max_recall
    

def main():

    import argparse

    hlp = "Compute some stats about alignments."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument("--alignments-path", required=True,
        help="Path to token alignments.")
    parser.add_argument("--output-vocab", required=True,
        help="Location to write output vocab.")

    args = parser.parse_args()
    #vocab_in = read_vocab(args.input_vocab)
    vocab_out = read_vocab(args.output_vocab) 

    for split in ["training", "validation", "test"]:
        print("Split: {}".format(split))
        data_path = os.path.join(args.data_path, args.corpus, split)
        alignments_path = os.path.join(
            args.alignments_path, args.corpus, split)
        
        collect_split_stats(data_path, alignments_path, vocab_out)


if __name__ == "__main__":
    main()
