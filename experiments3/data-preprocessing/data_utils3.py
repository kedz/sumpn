from collections import defaultdict

class Token:
    def __init__(self, token, lemma, pos, ne, ents):
        self.token = token
        self.lemma = lemma
        self.pos = pos
        self.ne = ne
        self.ents = frozenset(ents)

    def __str__(self):
        return self.token

    def lower(self):
        return self.token.lower()

    def __eq__(self, o):
        if type(self) == type(o):
            return self.token == o.token and self.lemma == o.lemma \
                    and self.pos == o.pos and self.ne == o.ne
        else:
            return self.token == o

class Document:
    def __init__(self, filename, url, title, highlights, sentences, input2ents,
                 highlight2ents):
        self.filename = filename
        self.url = url
        self.title = title
        self.highlights = highlights
        self.sentences = sentences
        self.input2ents = input2ents
        self.highlight2ents = highlight2ents

def read_line_document(line):
    filename, url, title, hl, sents, coref = line.split("\t")
    coref = coref.strip()
    filename = filename.strip()

    input2ents = defaultdict(lambda: defaultdict(set))
    highlight2ents = defaultdict(lambda: defaultdict(set))
    loc2ents = defaultdict(set)
    if len(coref) > 0:
        for c, chain in enumerate(coref.split(" "), 1):
            ent_str = "ent{}".format(c)
            for mention in chain.split("|"):
                src, sent, start, end = mention.split("-")
                sent = int(sent)
                start = int(start)
                end = int(end)
                
                if src == "i":
                    input2ents[sent][ent_str].add((ent_str, start, end))
                else:
                    highlight2ents[sent][ent_str].add((ent_str, start, end))
                for l in xrange(start, end):
                    loc2ents[(src, sent, l)].add((ent_str, start, end))

    highlights = list()
    for h, hl in enumerate(hl.split("|")):
        tokens = list()
        for t, token in enumerate(hl.split()):
            word, lemma, pos, ne = token.split("/")
            ents = loc2ents.get(("h", h, t), set())
            tokens.append(Token(word, lemma, pos, ne, ents))
        highlights.append(tokens)
    
    sentences = list()
    for s, sent in enumerate(sents.split("|")):
        tokens = list()
        for t, token in enumerate(sent.split()):
            word, lemma, pos, ne = token.split("/")
            ents = loc2ents.get(("i", s, t), set())
            tokens.append(Token(word, lemma, pos, ne, ents))
        sentences.append(tokens)

    return Document(filename, url, title, highlights, sentences, 
        input2ents, highlight2ents)
