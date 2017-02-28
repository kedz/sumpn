from __future__ import division, print_function
import sys
import os 
import re
from collections import defaultdict
from data_utils2 import read_line_document

ents = set(['ORGANIZATION', 'LOCATION', 'PERSON'])

def main():

    import argparse

    hlp = "Compute input vocabulary."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument("--documents", required=True,
        help="Path to preprocessed documents.")
    parser.add_argument("--output", required=True,
        help="Path to write vocab.")
    parser.add_argument("--size", required=True, type=int,
        help="Number of most frequent vocab words to keep.")
    parser.add_argument("--special", nargs="+", 
        default=["<E>", "<D>", "<S>", "<B>", "__UNK__", "__ENT__"])

    args = parser.parse_args()

    assert(args.size > 0)
    counts = defaultdict(int)

    vocab_dir = os.path.dirname(args.output)
    if vocab_dir != "" and not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    with open(args.documents, "r") as f:
        for i, line in enumerate(f, 1):
            sys.stdout.write("\rRead {:7d} documents".format(i))
            sys.stdout.flush()
            doc = read_line_document(line)
            for tokens in doc.highlights:
                for token in tokens:
                    if token.ne not in ents:
                        counts[re.sub(r"\d", "D", token.lower())] += 1
    
    counts = counts.items()
    counts.sort(key=lambda x: x[1], reverse=True)
    vocab = args.special + [w for w, c in counts[:args.size]]

    with open(args.output, "w") as f:
        f.write("\n".join(vocab)) 
   
if __name__ == "__main__":
    main()
