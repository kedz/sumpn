from __future__ import print_function

import json
from flask import Flask, render_template

from data_utils import read_vocab_ids

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize model predictions.')
    parser.add_argument('--json', required=True, help="Model json data.")
    parser.add_argument('--input-vocab', required=True, help="Input vocab.")
    parser.add_argument('--output-vocab', required=True, help="Output vocab.")

    args = parser.parse_args()
    
    with open(args.json, "r") as f:
        model_data = json.load(f)

    id2vocab_in, vocab2id_in = read_vocab_ids(args.input_vocab)
    id2vocab_out, vocab2id_out = read_vocab_ids(args.output_vocab)

    app = Flask(__name__)
    app.config["MODEL_DATA"] = model_data
    app.config["INPUT_VOCAB"] = (id2vocab_in, vocab2id_in)
    app.config["OUTPUT_VOCAB"] = (id2vocab_out, vocab2id_out)

    #app.config["ALIGN_PATHS"] = align_paths

    @app.route('/example/<example>')
    def display_example(example):

        example = int(example)

        data = app.config["MODEL_DATA"][example]
        max_steps = len(data["plates"][0]["steps"])

        id2vocab_in, vocab2id_in = app.config["INPUT_VOCAB"]
        id2vocab_out, vocab2id_out = app.config["OUTPUT_VOCAB"]

        return render_template("model-viewer.html", 
            example=example,
            max_steps=max_steps,
            input_vocab=json.dumps(id2vocab_in),
            output_vocab=json.dumps(id2vocab_out),
            plates=json.dumps(data["plates"]),
            output=json.dumps(data["output"]))
            #backbone_prediction_plate=json.dumps(bb_pred_plate),
            #support_prediction_plate=json.dumps(sp_pred_plate))

    app.run(port=8080, debug=True)
