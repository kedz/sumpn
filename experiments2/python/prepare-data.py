from __future__ import division, print_function
import os
import sys
import re
import yaml
from data_utils2 import read_line_document, read_vocab
from itertools import izip

ents = set(['ORGANIZATION', 'LOCATION', 'PERSON'])

def process_example(doc, alignments, vocab2id_in, id2vocab_in, 
                    vocab2id_out, id2vocab_out, ent_mode):

    sent2token_ids = list()
    sent2pretty_tokens = list()
    sent2tokens = list()

    id = 0
    for sent in doc.sentences:
        token_ids = list()
        pretty_tokens = [t.lower() for t in sent]
        if ent_mode == "1-tag":
            pp_tokens = [re.sub(r"\d", "D", t.lower()) 
                         if t.ne not in ents else "__ENT__" 
                         for t in sent]
        else:
            pp_tokens = [re.sub(r"\d", "D", t.lower()) 
                         if t.ne not in ents else "__{}__".format(t.ne[:3])
                         for t in sent]

        for token in pretty_tokens:
            token_ids.append(id)
            id += 1

        sent2token_ids.append(token_ids)
        sent2pretty_tokens.append(pretty_tokens)
        sent2tokens.append(pp_tokens)

    lines = list()
    for hl, (bb, sp, alignment) in izip(doc.highlights[:3], alignments[:3]):
        #print(" ".join([h.lower() for h in hl]))
        #print(bb, sp)
        if ent_mode == "1-tag":
            hl_tokens = [re.sub(r"\d", "D", t.lower()) 
                         if t.ne not in ents else "__ENT__" for t in hl]
        else:
            hl_tokens = [re.sub(r"\d", "D", t.lower()) 
                         if t.ne not in ents else "__{}__".format(t.ne[:3]) 
                         for t in hl]

        if bb == None: continue
        token_ids_flat = list(["<S>"])
        token_ids_flat.extend(sent2token_ids[bb])
        pretty_tokens_flat = list(["<S>"])
        pretty_tokens_flat.extend(sent2pretty_tokens[bb])

        pretty_bb_tokens = list(["<S>"])
        pretty_bb_tokens.extend(sent2pretty_tokens[bb])
        tokens_flat = list(["<S>"])
        tokens_flat.extend(sent2tokens[bb])

        pretty_sp_tokens = list()
        for support in sp:
            token_ids_flat.append("<B>")
            token_ids_flat.extend(sent2token_ids[support])
            pretty_sp_tokens.append("<B>")
            pretty_sp_tokens.extend(sent2pretty_tokens[support])
            pretty_tokens_flat.append("<B>")
            pretty_tokens_flat.extend(sent2pretty_tokens[support])
            tokens_flat.append("<B>")
            tokens_flat.extend(sent2tokens[support])

#        print(pretty_tokens_flat)
#        print(tokens_flat)


        relative_alignments = list()
        for i, a in enumerate(alignment):
            if a > -1:
                index = token_ids_flat.index(a)
                #print(a, index, hl_tokens[i], pretty_tokens_flat[index])
                relative_alignments.append(index + len(id2vocab_out))
            else:
                if hl_tokens[i] in vocab2id_out:
                    relative_alignments.append(vocab2id_out[hl_tokens[i]])
                else:
                    relative_alignments.append(vocab2id_out["__UNK__"])

#        print(relative_alignments)

        backbone_data_items = list()
        backbone_data_items.append(vocab2id_in.get("<E>"))
        for token in sent2tokens[bb]:
            backbone_data_items.append(
                vocab2id_in.get(token, vocab2id_in["__UNK__"]))
        backbone_data_str = " ".join(str(i) for i in backbone_data_items)

        support_data_items = list()

        for support in sp:
            support_data_items.append(vocab2id_in["<B>"])
            for token in sent2tokens[support]:
                support_data_items.append(
                    vocab2id_in.get(token, vocab2id_in["__UNK__"]))
        support_data_items.append(vocab2id_in["<B>"])

        support_data_str = " ".join(str(i) for i in support_data_items)
  
        relative_alignments = [vocab2id_out["<D>"]] + \
                relative_alignments + [vocab2id_out["<S>"]]
        target_data_str = " ".join(str(i) for i in relative_alignments)

        line = ''.join([backbone_data_str, " | ",
                        support_data_str, " | ",
                        target_data_str, " | ",
                        " ".join(pretty_bb_tokens), " | ",
                        " ".join(pretty_sp_tokens), " | ",
                        doc.filename,
                        "\n"])
#        print(line)
#
#        for bbi in backbone_data_items:
#            print(id2vocab_in[bbi])
#        for bbi in support_data_items:
#            print(id2vocab_in[bbi])
#        for r in relative_alignments:
#            if r >= 1006:
#                print(pretty_tokens_flat[r - 1006])
#            else:
#                print(id2vocab_out[r])
        lines.append(line)

    return lines

def main():

    import argparse

    help_msg = "Format data for sentence generation task using neural " \
            + "network implemented in Torch/Lua."
    parser = argparse.ArgumentParser(help_msg)
    parser.add_argument('--documents', required=True, 
        help="Path to preprocessed data.")
    parser.add_argument('--alignments', required=True, 
        help="Path to alignments data.")
    parser.add_argument('--output', required=True,
        help="File to write data.")
    parser.add_argument('--input-vocab', required=True,
        help="Path to input vocab.")
    parser.add_argument('--output-vocab', required=True,
        help="Path to output vocab.")
    parser.add_argument('--entity-mode', required=True,
        choices=["1-tag", "3-tags"])

    args = parser.parse_args()
     
    output_dir = os.path.dirname(args.output)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Reading alignments from {} ...".format(args.alignments))

    name2alignments = dict()
    with open(args.alignments, "r") as f:
        for i, (filename, alignments) in enumerate(yaml.load_all(f), 1):
            sys.stdout.write("\rRead {:7d} alignments".format(i))
            sys.stdout.flush()
            name2alignments[filename] = alignments
    print("")

    print("Reading input vocab from {} ...".format(args.input_vocab))
    id2vocab_in, vocab2id_in = read_vocab(args.input_vocab)

    print("Reading output vocab from {} ...".format(args.output_vocab))
    id2vocab_out, vocab2id_out = read_vocab(args.output_vocab)

    print("Reading documents from {} ...".format(args.documents))
    print("Writing data to {} ...".format(args.output))

    with open(args.documents, "r") as f, open(args.output, "w") as o:
        for i, line in enumerate(f, 1):
            sys.stdout.write("\rRead {:7d} documents".format(i))
            sys.stdout.flush()
            doc = read_line_document(line)
            if doc.filename not in name2alignments:
                print("\nSkipping {}, no alignment found.".format(
                    doc.filename))
                continue
            alignments = name2alignments[doc.filename]
            dls = process_example(doc, alignments, vocab2id_in, id2vocab_in, 
                vocab2id_out, id2vocab_out, args.entity_mode)
            for dl in dls: o.write(dl)
    print("")

if __name__ == "__main__":
    main()
