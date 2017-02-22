from __future__ import print_function
import argparse
from data_utils2 import read_line_document
import yaml
import json
import sys

from flask import Flask, render_template

def load_documents(documents_path):

    name2doc = dict()
    docs = list()
    print("Loading documents from {} ...".format(documents_path))
    with open(documents_path, "r") as f:
        for i, line in enumerate(f, 1):
            sys.stdout.write("{:10d} documents read.\r".format(i))
            sys.stdout.flush()
            doc = read_line_document(line)
            name2doc[doc.filename] = doc
            docs.append(doc)
            if len(docs) == 10: break
    print("") 
    return docs, name2doc

def load_alignments(alignments_path):
    name2alignments = dict()
    print("Loading alignments from {} ...".format(alignments_path))
    with open(alignments_path, "r") as f:
        for i, (filename, alignment) in enumerate(yaml.load_all(f), 1):
            sys.stdout.write("{:10d} alingments read.\r".format(i))
            sys.stdout.flush()
            name2alignments[filename] = alignment
            if len(name2alignments) == 10: break
    print("") 
    return name2alignments

def check_alignments(docs, name2alignments):
    print("Checking alignments...")
    good_docs = list()
    for doc in docs:
        if doc.filename in name2alignments:
            good_docs.append(doc)
        else:
            print("Document {} has no alignment!".format(doc.filename))
    return good_docs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Visualize alignments.')
    parser.add_argument('--documents', required=True, 
        help="Path to preprocessed content file.")
    parser.add_argument('--alignments', required=True, 
        help="Path to alignments data.")

    args = parser.parse_args()

    docs, name2doc = load_documents(args.documents)
    name2alignments = load_alignments(args.alignments)
    docs = check_alignments(docs, name2alignments)


    app = Flask(__name__)
    app.config["DOCS"] = docs
    app.config["ALIGNMENTS"] = name2alignments

    @app.route('/example/<example>')
    def display_example(example):
        example = int(example)

        doc = app.config["DOCS"][example]
        bb_sp_a = app.config["ALIGNMENTS"][doc.filename]
        doc_tokens = [[t.token for t in s] for s in doc.sentences]
        highlights = [[t.token for t in s] for s in doc.highlights]

        alignments = [a for _, _, a in bb_sp_a] 

        i = 0
        doc_token_ids = list()
        for tokens in doc_tokens:
            token_ids = list()
            for token in tokens:
                token_ids.append(i)
                i += 1
            doc_token_ids.append(token_ids)

        return render_template("alignment-viewer.html",
            doc_tokens=doc_tokens,
            highlights=highlights,
            alignments=alignments
            )


    app.run(host="0.0.0.0", port=8080, debug=True)
