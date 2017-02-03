from __future__ import division, print_function
import sys
import os
from data_utils import read_document

def test_read(data_path):

    doc_paths = [os.path.join(data_path, file) 
            for file in os.listdir(data_path)]
    doc_paths.sort()
    num_docs = len(doc_paths)


    bad_paths = list()

    for i, doc_path in enumerate(doc_paths, 1):
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, num_docs, 100 * i / num_docs))
        sys.stdout.flush()
        #try:
        try: 
            read_document(doc_path)
        except ValueError, e:
            bad_paths.append((str(e), doc_path))
    print("")

    num_bad = len(bad_paths)
    print("Found {:d} bad paths ( {:6.3f}% )".format(
        num_bad, 100 * num_bad / num_docs))
    return bad_paths

def main():

    import argparse

    hlp = "Compute some stats about the corpus."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument('--output', required=True, 
        help="Location to write bad paths.")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w") as f:
        for split in ["training", "validation", "test"]:
            print("Split: {}".format(split))

            data_path = os.path.join(args.data_path, args.corpus, split)
            bad_paths = test_read(data_path)
        
            for msg, path in bad_paths:
                f.write("{}\t{}\n".format(msg, path))


if __name__ == "__main__":
    main()
