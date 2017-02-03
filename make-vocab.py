from __future__ import division, print_function
import sys
import os 
from collections import defaultdict
import re
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf8')
from unidecode import unidecode
from data_utils import read_document, preprocess_tokens

def get_token_counts(data_path, max_sent, max_highlight):

    doc_paths = [os.path.join(data_path, file) 
            for file in os.listdir(data_path)]

    num_docs = len(doc_paths)
    counts_inp = defaultdict(int)
    counts_hl = defaultdict(int)

    for i, doc_path in enumerate(doc_paths, 1):
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, num_docs, 100 * i / num_docs))
        sys.stdout.flush()
        doc = read_document(doc_path)

        for sent in doc["sentences"][:max_sent]:
            tokens = preprocess_tokens(sent["tokens"], doc["entities"])
            for token in tokens: counts_inp[token] += 1

        for sent in doc["highlights"][:max_highlight]:
            tokens = preprocess_tokens(sent["tokens"], doc["entities"])
            for token in tokens: counts_hl[token] += 1

    return counts_inp, counts_hl

def compute_coverage_stats(vocab, counts):
    total = sum(tc[1] for tc in counts.items())
    ents = counts["__ENTITY__"]
    common = sum(tc[1] for tc in counts.items() if tc[0] in vocab)
    rare = total - ents - common
    return common, ents, rare, total

def print_stats(stats, split, source, output):
    common, ents, rare, total = stats
    output.write("\n")
    output.write("{}:\n".format(split))
    output.write("   {}:\n".format(source))
    output.write("       common   {:>15} ( {:6.2f}% )\n".format(
        locale.format("%d", common, grouping=True), 100 * common / total))
    output.write("       entities {:>15} ( {:6.2f}% )\n".format(
        locale.format("%d", ents, grouping=True), 100 * ents / total))
    output.write("       rare     {:>15} ( {:6.2f}% )\n".format(
        locale.format("%d", rare, grouping=True), 100 * rare / total))
    output.write("      =====================================\n")
    output.write("       total    {:>15}\n".format(
        locale.format("%d", total, grouping=True)))


def main():

    import argparse

    hlp = "Compute some stats about the corpus."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument('--max-input-sent', required=True, type=int,
        help="Maximum number of sentences to read from input.")
    parser.add_argument('--max-highlight', required=True, type=int,
        help="Maximum number of highlights to read.")
    parser.add_argument("--input-vocab", required=True,
        help="Location to write input vocab.")
    parser.add_argument("--output-vocab", required=True,
        help="Location to write output vocab.")
    parser.add_argument("--input-vocab-size", required=True, type=int,
        help="Size of input vocab.")
    parser.add_argument("--output-vocab-size", required=True, type=int,
        help="Size of output vocab.")
    parser.add_argument("--stats", required=True,
        help="Location to write vocab stats.")

    args = parser.parse_args()

    inp_vocab_dir = os.path.dirname(args.input_vocab)
    if inp_vocab_dir != "" and not os.path.exists(inp_vocab_dir):
        os.makedirs(inp_vocab_dir)

    out_vocab_dir = os.path.dirname(args.output_vocab)
    if out_vocab_dir != "" and not os.path.exists(out_vocab_dir):
        os.makedirs(out_vocab_dir)

    stats_dir = os.path.dirname(args.stats)
    if stats_dir != "" and not os.path.exists(stats_dir):
        os.makedirs(stats_dir)




    path_train = os.path.join(args.data_path, args.corpus, "training")
    print("reading counts from {} ...".format(path_train))
    counts_inp_train, counts_hl_train = get_token_counts(
        path_train, args.max_input_sent, args.max_highlight)

    path_dev = os.path.join(args.data_path, args.corpus, "validation")
    print("\nreading counts from {} ...".format(path_dev))
    counts_inp_dev, counts_hl_dev = get_token_counts(
        path_dev, args.max_input_sent, args.max_highlight)

    path_test = os.path.join(args.data_path, args.corpus, "test")
    print("\nreading counts from {} ...".format(path_test))
    counts_inp_test, counts_hl_test = get_token_counts(
        path_test, args.max_input_sent, args.max_highlight)

    print("")

    vocab_inp_counts = [tc for tc in counts_inp_train.items()
                        if tc[0] != "__ENTITY__"]
    vocab_inp_counts.sort(key=lambda x: x[1], reverse=True)
    vocab_inp = [tc[0] for tc in vocab_inp_counts][:args.input_vocab_size]

    vocab_hl_counts = [tc for tc in counts_hl_train.items() 
                        if tc[0] != "__ENTITY__"]
    vocab_hl_counts.sort(key=lambda x: x[1], reverse=True)
    vocab_hl = [tc[0] for tc in vocab_hl_counts][:args.output_vocab_size]

    with open(args.input_vocab, "w") as f:
        f.write("\n".join(vocab_inp))

    with open(args.output_vocab, "w") as f:
        f.write("\n".join(vocab_hl))

    vocab_inp = set(vocab_inp)
    vocab_hl = set(vocab_hl)

    with open(args.stats, "w") as f:
        stats_inp_train = compute_coverage_stats(vocab_inp, counts_inp_train)
        print_stats(stats_inp_train, "train", "input", f)

        stats_inp_dev = compute_coverage_stats(vocab_inp, counts_inp_dev)
        print_stats(stats_inp_dev, "dev", "input", f)

        stats_inp_test = compute_coverage_stats(vocab_inp, counts_inp_test)
        print_stats(stats_inp_test, "test", "input", f)

        stats_hl_train = compute_coverage_stats(vocab_hl, counts_hl_train)
        print_stats(stats_hl_train, "train", "highlight", f)

        stats_hl_dev = compute_coverage_stats(vocab_hl, counts_hl_dev)
        print_stats(stats_hl_dev, "dev", "highlight", f)

        stats_hl_test = compute_coverage_stats(vocab_hl, counts_hl_test)
        print_stats(stats_hl_test, "test", "highlight", f)

   
if __name__ == "__main__":
    main()
