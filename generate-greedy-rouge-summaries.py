from __future__ import division
import sys
import os
import random
from itertools import product
from tempfile import NamedTemporaryFile
import subprocess
from data_utils import read_document, get_document_paths, replace_entities
from multiprocessing import Pool

def build_rouge_params(rouge_settings):

    rouge_home = os.path.abspath(rouge_settings["path"])
    rouge_path = os.path.join(rouge_home, "ROUGE-1.5.5.pl")
    data_path = os.path.join(rouge_home, "data")

    params = ["perl", rouge_path, "-e", data_path, "-d", "-s",
              "-b", str(rouge_settings['size'])]
   
    if rouge_settings["order"] != "L":
       params.extend(["-x", "-n", rouge_settings["order"]])

    params.extend(["-z", "SPL"])
   
    return params

def get_input_sentences(doc, sent_limit):

    sentences = list()
    for sentence in doc["sentences"][:25]:
        sent_tokens = replace_entities(sentence["tokens"], doc["entities"])
        sentences.append(" ".join(sent_tokens))
    return sentences
 
def get_reference_file(doc):
    highlights = list()
    for highlight in doc["highlights"]:
        highlight_tokens = replace_entities(highlight["tokens"], doc["entities"])
        highlights.append(" ".join(highlight_tokens))
    ref_text = "\n".join(highlights)
    ref_file = NamedTemporaryFile("w", delete=False)
    ref_file.write(ref_text)
    ref_file.close()
    return ref_file

def build_summary(doc, rouge_settings):

    params = build_rouge_params(rouge_settings)
    match_pattern = "X ROUGE-{} Eval".format(rouge_settings["order"])

    ref_file = get_reference_file(doc)

    input_sentences = get_input_sentences(doc, 25)
    n_inputs = len(input_sentences)
    input_ids = [i for i in xrange(n_inputs)]

    candidate_files = [NamedTemporaryFile("w", delete=False) 
                       for i in xrange(n_inputs)]
    config_lines = ["{} {}".format(cf.name, ref_file.name)
                    for cf in candidate_files]

    config_file = NamedTemporaryFile("w", delete=False)
    
    greedy_summary = ""
    greedy_score = 0

    for z in range(n_inputs):

        cfg_text = "\n".join(config_lines)
        config_file.truncate(len(cfg_text))
        config_file.seek(0)
        config_file.write(cfg_text)
        config_file.flush()

        for i in xrange(len(config_lines)):
            input_id = input_ids[i]
            sum = "{}{}\n".format(greedy_summary, input_sentences[input_id])
            cf = candidate_files[i]
            cf.truncate(len(sum))
            cf.seek(0)
            cf.write(sum)
            cf.flush()

        output = subprocess.check_output(params + [config_file.name])
        i = 0
        max_score = greedy_score
        max_id = None
        for line in output.split("\n"):
            if line.startswith(match_pattern):

                score = float(line.split()[4][2:])
                if score > max_score:
                    max_score = score
                    max_id = i
                i += 1
        if max_id is not None:
            greedy_score = max_score
            greedy_summary = "{}{}\n".format(
                greedy_summary, input_sentences[max_id])
            input_ids.pop(max_id)
            config_lines = config_lines[:-1]
        else:
            break
    
    for cf in candidate_files:
        cf.close()
        os.remove(cf.name)
    os.remove(ref_file.name)
    config_file.close()
    os.remove(config_file.name)

    return greedy_summary.strip()

def process_document(args):
    document_path, output_dir, rouge_settings = args

    output_path = os.path.join(output_dir, os.path.split(document_path)[1])
    doc = read_document(document_path)
    summary_text = build_summary(doc, rouge_settings)
    with open(output_path, "w") as f:
        f.write(summary_text)

def main():

    import argparse

    hlp = "Create summaries by greedily optimizing ROUGE."

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
    parser.add_argument('--rouge', required=True,
        help="ROUGE root directory")
    parser.add_argument('--order', required=True, choices=["1", "2", "L"],
        help="ROUGE ngram order or longest common subsequence.")
    parser.add_argument('--size', required=True, type=int, 
        help="ROUGE byte limit to use.")

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


    rouge_settings = {"path": args.rouge, 
                      "order": args.order, 
                      "size": args.size} 

    document_paths = get_document_paths(input_path, args.samples, args.random)    
    n_docs = len(document_paths)

    args = [(dp, args.output, rouge_settings) for dp in document_paths]
    pool = Pool(processes=12)
    
    for i, d in enumerate(pool.imap_unordered(process_document, args), 1):
        sys.stdout.write("\r {:7.4f}% ".format(100 * i / n_docs))
        sys.stdout.flush()
        #process_document(document_path, args.output, rouge_settings)
    print 
        
if __name__ == "__main__":
    main()
