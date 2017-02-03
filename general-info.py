from __future__ import division, print_function
import sys
import os 
import re
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
from data_utils import read_document, replace_entities

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
            tokens = replace_entities(sent["tokens"], doc["entities"])
            num_input_tokens.append(len(tokens))
            doc_i_len_tokens += len(tokens)
            if s < 25:
                doc_i_len_tokens_trunc += len(tokens)

        doc_len_tokens.append(doc_i_len_tokens)
        doc_len_tokens_trunc.append(doc_i_len_tokens_trunc)

        ref_i_len_tokens = 0
        hl_tokens = list()
        hl_tokens_flat = list()
        for highlight in doc["highlights"]:
            tokens = replace_entities(highlight["tokens"], doc["entities"])
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

        row_data = [np.mean(data), np.median(data), 
                    np.std(data), np.max(data)]
        row_data.extend(np.percentile(data, percentiles))
        return row_data
         
    df_data = list()
    df_data.append(make_data_row(num_inputs))
    df_data.append(make_data_row(doc_len_tokens))
    df_data.append(make_data_row(doc_len_tokens_trunc))
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
        [("", "mean"), ("", "median"), ("", "std"), ("", "max")] + \
        [("Percentile", "{}th".format(p)) for p in percentiles])

    index = ["inp. len. (sents.)", "inp. len. (tok.)",
             "inp. len. trunc25sent (tok.)", "inp. sent. len. (toks.)",  
             "hl. len. (sents.)", "hl. len. (tok.)", "hl. sent. len. (toks.)",
             "ref[:75] len. (sents.)", "ref[:75] len. (tok.)",
             "ref[:250] len. (sents.)", "ref[:250] len. (tok.)",
             "ref[:+inf] len. (sents.)", "ref[:+inf] len. (tok.)"] 


    df = pd.DataFrame(df_data, columns=columns, index=index)
    df_str_lines = str(df).split("\n")

    print("\n".join(df_str_lines[:2]) + "\n")
    print("\n".join(df_str_lines[2:6]) + "\n")
    print("\n".join(df_str_lines[6:9]) + "\n")
    print("\n".join(df_str_lines[9:11]) + "\n")
    print("\n".join(df_str_lines[11:13]) + "\n")
    print("\n".join(df_str_lines[13:15]) + "\n")


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
