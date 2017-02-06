from __future__ import division
import os
import sys
import random
from data_utils import read_document, replace_entities, get_document_paths

def build_summary(doc, lead):

    sents = list()
    for sent in doc["sentences"][:lead]:
        sent_tokens = replace_entities(sent["tokens"], doc["entities"])
        sents.append(" ".join(sent_tokens))
    return "\n".join(sents)

def process_document(document_path, output_dir, lead):

    output_path = os.path.join(output_dir, os.path.split(document_path)[1])
    doc = read_document(document_path)
    summary_text = build_summary(doc, lead)
    with open(output_path, "w") as f:
        f.write(summary_text)

def main():

    import argparse

    hlp = "Create reference summaries using highlights."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--split', required=True, help="Data split to use.",
        choices=["train", "dev", "test"])
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument('--samples', default=-1, type=int,
        help="Number of samples. -1 indicates all data.")
    parser.add_argument('--random', default=True, type=bool,
        help="Shuffle data.")
    parser.add_argument('--output', required=True,
        help="Output directory to write summaries.")
    parser.add_argument('--seed', default=1986, type=int,
        help="Random seed if random is true.")
    parser.add_argument('--lead', required=True, type=int,
        help="Number of lead sentences to use.")

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
    n_docs = len(document_paths)

    for i, document_path in enumerate(document_paths, 1):
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, n_docs, 100 * i / n_docs))
        sys.stdout.flush()
        process_document(document_path, args.output, args.lead)
        
if __name__ == "__main__":
    main()
