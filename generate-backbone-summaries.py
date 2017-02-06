from __future__ import division
import sys
import os
import random
from data_utils import get_document_paths, read_document, replace_entities
from itertools import izip
import yaml

def build_summary(doc_path, align_path, summary_path):

    backbones = []
    used = set()
    with open(align_path, "r") as f:
        alignments = yaml.load(f)
        for backbone, support, ta in alignments:
            if backbone != None and backbone not in used:
                backbones.append(backbone)
                used.add(backbone)

    doc = read_document(doc_path)

    lines = list()
    for b in backbones:
        tokens = doc["sentences"][b]["tokens"]
        sent_str = " ".join(replace_entities(tokens, doc["entities"]))
        lines.append(sent_str)

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

def main():

    import argparse

    hlp = "Create oracle extractive summaries using sentence fusion " \
          "backbone\n\tfinding algorithm."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--split', required=True, help="Data split to use.",
        choices=["train", "dev", "test"])
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument("--alignments-path", required=True,
        help="Path to token alignments.")
    parser.add_argument('--samples', default=-1, type=int,
        help="Number of samples. -1 indicates all data.")
    parser.add_argument('--random', default=False, action="store_true",
        help="Shuffle data.")
    parser.add_argument('--output', required=True,
        help="Output directory to write summaries.")
    parser.add_argument('--seed', default=1986, type=int,
        help="Random seed if random is true.")

    args = parser.parse_args()
     
    arg2split = {"test": "test", "train": "training", "dev": "validation"}
    split = arg2split[args.split]

    input_dir = os.path.join(args.data_path, args.corpus, split)
    if not os.path.exists(input_dir):
        raise Exception("Path to data ({:s}) does not exist.".format(
            input_dir))
    alignments_dir = os.path.join(args.alignments_path, args.corpus, split)
    if not os.path.exists(alignments_dir):
        raise Exception("Path to data ({:s}) does not exist.".format(
            alignments_dir))
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    random.seed(args.seed)

    doc_paths = get_document_paths(input_dir, args.samples, args.random)
    align_paths = get_document_paths(alignments_dir, 
        args.samples, args.random)
    
    n_docs = len(doc_paths)

    path_iter = izip(doc_paths, align_paths)
    for i, (doc_path, align_path) in enumerate(path_iter, 1):
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, n_docs, 100 * i / n_docs))
        sys.stdout.flush()

        if not os.path.basename(doc_path) == os.path.basename(align_path):
            raise Exception(
                 "Alignments directory does not contain one file for every " \
                 "file in data path.")
        output_path = os.path.join(args.output, os.path.basename(doc_path))

        build_summary(doc_path, align_path, output_path)

    print

if __name__ == "__main__":
    main()
