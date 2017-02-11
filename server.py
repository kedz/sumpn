from __future__ import print_function

import os

from data_utils import get_document_paths, read_document, replace_entities
from itertools import izip
import yaml

from flask import Flask, render_template
import json

#        for i, val in enumerate(hl):
#            if val == -99:
#                if highlight_tokens[h][i] in stopwords:
#                    highlight_alignments[h][i] = -98
#                elif highlight_tokens[h][i] in top5kvocab:
#                    highlight_alignments[h][i] = -97
#
#    return render_template("default.html", doc_tokens=doc_tokens,
#            highlights=highlight_tokens, alignments=highlight_alignments,
#            alignments_json=json.dumps(highlight_alignments),
#            backbone_ids=backbone_ids)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize copy sequence alignment.')
    parser.add_argument('--split', required=True, help="Data split to use.",
        choices=["train", "dev", "test"])
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")
    parser.add_argument("--alignments-path", required=True,
        help="Path to token alignments.")

    args = parser.parse_args()
     
    arg2split = {"test": "test", "train": "training", "dev": "validation"}
    split = arg2split[args.split]

    doc_dir = os.path.join(args.data_path, args.corpus, split)
    align_dir = os.path.join(args.alignments_path, args.corpus, split)
    
    doc_paths = get_document_paths(doc_dir, -1, False)
    align_paths = get_document_paths(align_dir, -1, False)

    for doc_path, align_path in izip(doc_paths, align_paths):
        if not os.path.basename(doc_path) == os.path.basename(align_path):
            raise Exception(
                 "Alignments directory does not contain one file for every " \
                 "file in data path.")
     


#    datasets = read_data(args.data) 
#
    
    app = Flask(__name__)
    app.config["DOC_PATHS"] = doc_paths
    app.config["ALIGN_PATHS"] = align_paths


    @app.route('/example/<example>')
    def display_example(example):

        example = int(example)

        doc_path = app.config["DOC_PATHS"][example]
        align_path = app.config["ALIGN_PATHS"][example]
        doc = read_document(doc_path)
        doc_tokens = [replace_entities(s["tokens"], doc["entities"])
                      for s in doc["sentences"][:25]]
        highlight_tokens = [replace_entities(s["tokens"], doc["entities"])
                      for s in doc["highlights"][:4]]

        i=0

        doc_token_ids = list()
        for tokens in doc_tokens:
            token_ids = list()
            for token in tokens:
                token_ids.append(i)
                i += 1
            doc_token_ids.append(token_ids)


        backbone_ids = list()
        alignments = list()
        with open(align_path, "r") as f:
            #backbones, support, alignments 
            data = yaml.load(f)
            for backbone, support, alignment in data:
                if backbone != None:
                    backbone_ids.append(doc_token_ids[backbone])
                else:
                    backbone_ids.append(list())
                alignments.append(alignment)
        return render_template("default.html", doc_tokens=doc_tokens,
            highlights=highlight_tokens, alignments=alignments,
            alignments_json=json.dumps(alignments),
            backbone_ids=json.dumps(backbone_ids))



        #return "<br/>".join(" ".join(s) for s in doc_tokens) + "<br/><br/>" + "<br/>".join(" ".join(s) for s in highlight_tokens)

    app.run(port=8080, debug=True)
