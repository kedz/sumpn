from __future__ import print_function
import re
import json
import yaml
from flask import Flask, render_template

pattern = r'<p class="mol-para-with-font"><font>(.*?)</font>'
list_body_patt = r'<ul class="mol-bullets-with-font">(.*?)</ul>'
list_elem_patt = r'<li class=""><font><strong>(.*?)</strong>'
from HTMLParser import HTMLParser
h = HTMLParser()
a_patt = r"<a[^>]*>(.*?)</a>"

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Article extraction')
    parser.add_argument('--data', required=True, 
        help="Article content tsv file.")
    args = parser.parse_args()
    
    app = Flask(__name__)
    app.config["DATA"] = args.data

    @app.route('/example/<example>')
    def display_example(example):

        example = int(example)
        data = None
        with open(app.config["DATA"], "r") as f:
            for i, line in enumerate(f):
                if i == example:
                    data = line.strip()
        if data == None: return "BAD EXAMPLE"
        url, title, content = data.split("\t")
        content = content.replace("__NL__", "\n").replace("__TAB__", "\t")

        for match in re.findall(pattern, content, flags=re.DOTALL):
            print(re.sub(a_patt, r"\1", h.unescape(match)))
        print("\n\n\n")
        list_body = re.search(list_body_patt, content, flags=re.DOTALL)
        list_body = list_body.groups()[0]
        for match in re.findall(list_elem_patt, list_body, flags=re.DOTALL):
            print(h.unescape(match))

        return content

        #data = app.config["MODEL_DATA"][example]
        #max_steps = len(data["plates"][0]["steps"])

        #id2vocab_in, vocab2id_in = app.config["INPUT_VOCAB"]
        #id2vocab_out, vocab2id_out = app.config["OUTPUT_VOCAB"]

#        return render_template("model-viewer.html", 

    app.run(port=8081, debug=True)
