from data_utils import read_vocab_ids, read_document, replace_entities, preprocess_tokens, get_document_paths
import yaml
from itertools import izip

vocab_in_path = "data/input-vocab25k.txt"
vocab_out_path = "data/output-vocab1k.txt"
id2vocab_in, vocab2id_in = read_vocab_ids(vocab_in_path)
id2vocab_out, vocab2id_out = read_vocab_ids(vocab_out_path)

def process_example(doc_path, align_path):
    
    print doc_path
    doc = read_document(doc_path)

    sent2token_ids = list()
    sent2pretty_tokens = list()
    sent2tokens = list()

    id = 0
    for sent in doc["sentences"]:
        token_ids = list()
        pretty_tokens = replace_entities(sent["tokens"], doc["entities"])
        pp_tokens = preprocess_tokens(sent["tokens"], doc["entities"])
        for token in pretty_tokens:
            token_ids.append(id)
            #pretty_tokens.append(token)
            id += 1

        sent2token_ids.append(token_ids)
        sent2pretty_tokens.append(pretty_tokens)
        sent2tokens.append(pp_tokens)

    hl_tokens_pretty = replace_entities(doc["highlights"][0]["tokens"], doc["entities"])
    hl_tokens = preprocess_tokens(doc["highlights"][0]["tokens"], doc["entities"])

    with open(align_path, "r") as f:   
        backbone, supports, alignments = yaml.load(f)[0]
        
    token_ids_flat = list(["<S>"])
    token_ids_flat.extend(sent2token_ids[backbone])
    pretty_tokens_flat = list(["<S>"])
    pretty_tokens_flat.extend(sent2pretty_tokens[backbone])
    tokens_flat = list(["<S>"])
    tokens_flat.extend(sent2tokens[backbone])

    for support in supports:
        token_ids_flat.append("<B>")
        token_ids_flat.extend(sent2token_ids[support])
        pretty_tokens_flat.append("<B>")
        pretty_tokens_flat.extend(sent2pretty_tokens[support])
        tokens_flat.append("<B>")
        tokens_flat.extend(sent2tokens[support])

    relative_alignments = list()
    for i, a in enumerate(alignments):
        if a > -1:
            index = token_ids_flat.index(a)
            relative_alignments.append(index)
        else:
            if hl_tokens[i] in vocab2id_out:
                relative_alignments.append(-1)
            else:
                relative_alignments.append(-99)
    
    print 
    print len(supports)
    print pretty_tokens_flat
    print hl_tokens_pretty
    print relative_alignments
    print [pretty_tokens_flat[a] if a > -1 else -1 for a in relative_alignments]
            
    print [a + len(vocab2id_out) if a > -1 else a for a in relative_alignments]


    relative_alignments = list()
    for i, a in enumerate(alignments):
        if a > -1:
            index = token_ids_flat.index(a)
            relative_alignments.append(index + len(id2vocab_out))
        else:
            if hl_tokens[i] in vocab2id_out:
                relative_alignments.append(vocab2id_out[hl_tokens[i]])
            else:
                relative_alignments.append(vocab2id_out["__UNK__"])
    print relative_alignments 


    backbone_data_items = list()
    backbone_data_items.append(vocab2id_in.get("<S>"))
    for token in sent2tokens[backbone]:
        backbone_data_items.append(vocab2id_in.get(token, vocab2id_in["__UNK__"]))
    backbone_data_str = " ".join(str(i) for i in backbone_data_items)


    print sent2tokens[backbone]
    print [vocab2id_in.get(token, vocab2id_in["__UNK__"]) for token in sent2tokens[backbone]]
    print backbone_data_str
    print

    support_data_items = list()

    for support in supports:
        print sent2tokens[support]
        print [vocab2id_in.get(token, vocab2id_in["__UNK__"]) for token in sent2tokens[support]]
        print
        support_data_items.append(vocab2id_in["<B>"])
        for token in sent2tokens[support]:
            support_data_items.append(
                vocab2id_in.get(token, vocab2id_in["__UNK__"]))
    support_data_items.append(vocab2id_in["<B>"])

    support_data_str = " ".join(str(i) for i in support_data_items)
  
    relative_alignments = [vocab2id_out["<D>"]] + relative_alignments + [vocab2id_out["<E>"]]
    target_data_str = " ".join(str(i) for i in relative_alignments)

    print "THEDATA"
    print "======="
    print backbone_data_str
    print support_data_str
    print target_data_str
    
    print
    print [id2vocab_in[i] for i in backbone_data_items]
    print [id2vocab_in[i] for i in support_data_items]
    print [i if i < len(id2vocab_out) else pretty_tokens_flat[i - len(id2vocab_out)]
           for i in relative_alignments]

    return " | ".join([backbone_data_str, support_data_str, target_data_str])



doc_path = "neuralsum/dailymail/validation/ffed632c10296c495418d01e089491588d0fe684.summary"
align_path = "data/alignments/dailymail/validation/ffed632c10296c495418d01e089491588d0fe684.summary"

doc_paths = get_document_paths("neuralsum/dailymail/validation")
align_paths = get_document_paths("data/alignments/dailymail/validation")

with open("lead-data.txt", "w") as f:
    for doc_path, align_path in izip(doc_paths[:10], align_paths[:10]):
        print doc_path
        f.write(process_example(doc_path, align_path) + "\n")

