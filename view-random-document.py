from __future__ import print_function
import os
import random
from data_utils import read_document, replace_entities, preprocess_tokens
import textwrap

def main():

    import argparse

    hlp = "View a random document"

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument('--split', required=True, help="Data split to use.",
        choices=["train", "dev", "test"])
    parser.add_argument('--replace-entities', default=False, 
        action="store_true")
    parser.add_argument('--pproc', default=False, 
        action="store_true")

    args = parser.parse_args()

    arg2split = {"test": "test", "train": "training", "dev": "validation"}
    split = arg2split[args.split]

    data_path = os.path.join(args.data_path, args.corpus, split)
    doc_paths = [os.path.join(data_path, file) 
                 for file in os.listdir(data_path)]
    doc_paths.sort()
    random.shuffle(doc_paths)

    doc = read_document(doc_paths[0])

    print("url")
    print("===")
    print(doc["url"])

    print("\nINPUT")
    print("=====")
    for s, sent in enumerate(doc["sentences"], 1):
        tokens = sent["tokens"]
        if args.pproc:
            tokens = preprocess_tokens(tokens, doc["entities"])
        if args.replace_entities:
            tokens = replace_entities(tokens, doc["entities"]) 
        sent_str = " ".join(tokens)
        line = "{}) [{}] {}".format(s, sent["score"], sent_str)
        print(textwrap.fill(line, subsequent_indent="   "))

    print("\nENTITIES")
    print("========")
    for id, entity in sorted(doc["entities"].items(), key=lambda x: x[0]):
        print("{:10} :: {}".format(id, entity))

    print("\nHIGHLIGHTS")
    print("==========")

    for s, sent in enumerate(doc["highlights"], 1):
        tokens = sent["tokens"]
        if args.pproc:
            tokens = preprocess_tokens(tokens, doc["entities"])
        if args.replace_entities:
            tokens = replace_entities(tokens, doc["entities"]) 
        sent_str = " ".join(tokens)
        line = "{}) {}".format(s, sent_str)
        print(textwrap.fill(line, subsequent_indent="   "))
    
if __name__ == "__main__":
    main()
