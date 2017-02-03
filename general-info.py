from __future__ import division, print_function
import sys
import os 
import re
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)

def preprocess_tokens(tokens, doc):

    pp_tokens = list()
    for token in tokens:
        token = doc["entities"].get(token, token) 
        token = re.sub(r"^\**(.*?)\**$", r"\1", token).lower()
        for st in token.split():
            pp_tokens.append(st)
    return pp_tokens

def read_document(path):
    with open(path, "r") as f:
        data = f.read()
        
        url, article, ref, entities = data.split("\n\n")
        sentences = list()
        highlights = list()
        entity_id2name = dict()

        for sent in article.split("\n"):
            sent, score = sent.split("\t\t\t")
            tokens = sent.split(' ')
            score = int(score)
            sentences.append(
                {"score": score, "tokens": tokens, "string": sent})

        for sent in ref.split("\n"):
            tokens = sent.split(' ')
            highlights.append({"tokens": tokens, "string": sent})

        for entity in entities.split("\n"):
            label, value = entity.split(":", 1)
            entity_id2name[label] = value

    return {"sentences": sentences, "highlights": highlights,
            "entities": entity_id2name, "url": url}


def collect_split_stats(data_path):
    
    doc_paths = [os.path.join(data_path, file) 
            for file in os.listdir(data_path)]

    num_docs = len(doc_paths)

    num_highlights = list()
    num_inputs = list()
    num_input_tokens = list()
    num_highlight_tokens = list()
    
    doc_len_tokens = list()
    doc_len_tokens_trunc = list()
    ref_len_tokens = list()


    num_ref_trunc75_tokens = list()
    num_ref_trunc250_tokens = list()
    num_ref_truncNA_tokens = list()
    num_ref_trunc75_sents = list()
    num_ref_trunc250_sents = list()
    num_ref_truncNA_sents = list()



    for i, doc_path in enumerate(doc_paths, 1):
        sys.stdout.write("\r {:d} / {:d} ( {:7.4f}% ) ".format(
            i, num_docs, 100 * i / num_docs))
        sys.stdout.flush()
        doc = read_document(doc_path)
        num_highlights.append(len(doc["highlights"]))
        num_inputs.append(len(doc["sentences"]))
        
        doc_i_len_tokens = 0
        doc_i_len_tokens_trunc = 0 

        for s, sent in enumerate(doc["sentences"]):
            tokens = preprocess_tokens(sent["tokens"], doc)
            num_input_tokens.append(len(tokens))
            doc_i_len_tokens += len(tokens)
            if s < 30:
                doc_i_len_tokens_trunc += len(tokens)

        doc_len_tokens.append(doc_i_len_tokens)
        doc_len_tokens_trunc.append(doc_i_len_tokens_trunc)

        ref_i_len_tokens = 0
        hl_tokens = list()
        hl_tokens_flat = list()
        for highlight in doc["highlights"]:
            tokens = preprocess_tokens(highlight["tokens"], doc)
            num_highlight_tokens.append(len(tokens))
            hl_tokens.append(tokens)
            hl_tokens_flat.extend(tokens)
            ref_i_len_tokens += len(tokens)

        ref_len_tokens.append(ref_i_len_tokens)

        ref_text = "\n".join([" ".join(tokens) for tokens in hl_tokens])
        ref_text_flat = " ".join(hl_tokens_flat)

        ref_trunc75 = ref_text[:75]
        ref_trunc75_flat = ref_text_flat[:75]
        num_ref_trunc75_tokens.append(len(ref_trunc75_flat.split()))
        num_ref_trunc75_sents.append(len(ref_trunc75.split("\n")))
        
        ref_trunc250 = ref_text[:250]
        ref_trunc250_flat = ref_text_flat[:250]
        num_ref_trunc250_tokens.append(len(ref_trunc250_flat.split()))
        num_ref_trunc250_sents.append(len(ref_trunc250.split("\n")))
 
        ref_truncNA = ref_text
        ref_truncNA_flat = ref_text_flat
        num_ref_truncNA_tokens.append(len(ref_truncNA_flat.split()))
        num_ref_truncNA_sents.append(len(ref_truncNA.split("\n")))
        
    sys.stdout.write("\n")
    sys.stdout.flush()

    percentiles = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    def make_data_row(data):

        row_data = [np.mean(data), np.median(data), np.std(data)]
        row_data.extend(np.percentile(data, percentiles))
        return row_data
         
    df_data = list()
    df_data.append(make_data_row(num_inputs))
    df_data.append(make_data_row(doc_len_tokens))
    df_data.append(make_data_row(num_input_tokens))

    df_data.append(make_data_row(num_highlights))
    df_data.append(make_data_row(ref_len_tokens))
    df_data.append(make_data_row(num_highlight_tokens))
    
    df_data.append(make_data_row(num_ref_trunc75_sents))
    df_data.append(make_data_row(num_ref_trunc75_tokens))
    df_data.append(make_data_row(num_ref_trunc250_sents))
    df_data.append(make_data_row(num_ref_trunc250_tokens))
    df_data.append(make_data_row(num_ref_truncNA_sents))
    df_data.append(make_data_row(num_ref_truncNA_tokens))


    columns = pd.MultiIndex.from_tuples(
        [("", "mean"), ("", "median"), ("", "std")] + \
        [("Percentile", "{}th".format(p)) for p in percentiles])

    index = ["inp. len. (sents.)", "inp. len. (tok.)", "inp. sent. len. (toks.)",  
             "hl. len. (sents.)", "hl. len. (tok.)", "hl. sent. len. (toks.)",
             "ref[:75] len. (sents.)", "ref[:75] len. (tok.)",
             "ref[:250] len. (sents.)", "ref[:250] len. (tok.)",
             "ref[:+inf] len. (sents.)", "ref[:+inf] len. (tok.)"] 


    df = pd.DataFrame(df_data, columns=columns, index=index)
    df_str_lines = str(df).split("\n")

    print("\n".join(df_str_lines[:2]) + "\n")
    for i in xrange(2, 8, 3):
        print("\n".join(df_str_lines[i:i+3]) + "\n")
    for i in xrange(8, len(df_str_lines), 2):
        print("\n".join(df_str_lines[i:i+2]) + "\n")


def main():

    import argparse

    hlp = "Compute some stats about the corpus."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--corpus', required=True, help="Corpus to use.",
        choices=["dailymail", "cnn"])
    parser.add_argument('--data-path', required=True, 
        help="Path to Cheng&Lapata data.")

    args = parser.parse_args()

    for split in ["training", "validation", "test"]:
        print("Split: {}".format(split))
        data_path = os.path.join(args.data_path, args.corpus, split)
        collect_split_stats(data_path)



if __name__ == "__main__":
    main()
