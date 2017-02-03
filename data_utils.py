import os
import re
import random
from unidecode import unidecode

def read_document(path):
    with open(path, "r") as f:

        utf_data = f.read().decode("utf-8")
        data = unidecode(utf_data)

        url, article, ref, entities = data.split("\n\n")
        sentences = list()
        highlights = list()
        entity_id2name = dict()

        for sent in article.split("\n"):
            sent, score = sent.split("\t\t\t")
            
            tokens = [re.sub(r"^\**(.*?)\**$", r"\1", token).lower()
                      for token in sent.split()]

            score = int(score)
            sentences.append(
                {"score": score, "tokens": tokens, "string": sent})

        for sent in ref.split("\n"):
            tokens = [re.sub(r"^\**(.*?)\**$", r"\1", token).lower()
                      for token in sent.split()]
            highlights.append({"tokens": tokens, "string": sent})

        for entity in entities.split("\n"):
            label, value = entity.split(":", 1)
            value_tokens = [re.sub(r"^\**(.*?)\**$", r"\1", token).lower()
                            for token in value.split()]

            entity_id2name[label] = " ".join(value_tokens)

    return {"sentences": sentences, "highlights": highlights,
            "entities": entity_id2name, "url": url}

def replace_entities(tokens, entities):

    repl_tokens = list()
    for token in tokens:
        nes = entities.get(token, None)
        if nes is not None:
            repl_tokens.extend(nes.split())
        else:
            repl_tokens.append(token)

    return repl_tokens

def preprocess_tokens(tokens, entities):
    pp_tokens = list()
    for token in tokens:
        nes = entities.get(token, None)
        if nes is not None:
            for st in nes.split():
                pp_tokens.append("__ENTITY__")
        else:        
            token = re.sub(r"\d", r"D", token)
            pp_tokens.append(token)

    return pp_tokens

def get_document_paths(directory, sample_size=-1, shuffle=False):
    filenames = os.listdir(directory)
    filenames.sort()
    paths = [os.path.join(directory, filename)
             for filename in filenames]
    if shuffle:
        random.shuffle(paths)
    if sample_size == -1:
        return paths
    elif sample_size > 0:
        return paths[:sample_size]
    else:
        raise ValueError(
           "Invalid sample size: {:d}. Must be positive or -1".format(
               sample_size))

unk_id = -98

sw_id = -99

stopwords = set(["a", "about", "above", "across", "after", "afterwards",
    "again", "against", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an",
    "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
    "anywhere", "are", "around", "as", "at", "back", "be", "became", "because",
    "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "computer",
    "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herse", "him", "himse", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itse", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myse", "name", "namely", "neither", "never",
    "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor",
    "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once",
    "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part", "per", "perhaps", "please",
    "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
    "serious", "several", "she", "should", "show", "side", "since", "sincere",
    "six", "sixty", "so", "some", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
    "ten", "than", "that", "the", "their", "them", "themselves", "then",
    "thence", "there", "thereafter", "thereby", "therefore", "therein",
    "thereupon", "these", "they", "thick", "thin", "third", "this", "those",
    "though", "three", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
    "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we",
    "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
    "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
    "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
    "whose", "why", "will", "with", "within", "without", "would", "yet", "you",
    "your", "yours", "yourself", "yourselves", ",", ".", "\'", "\"", ";", ":",
    "?", "'s", "'ve", "'d", "'ll", "'re", "n't", "-", "_", 'did'])


