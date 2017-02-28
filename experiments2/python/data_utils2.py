
class Token:
    def __init__(self, token, pos, ne):
        self.token = token
        self.pos = pos
        self.ne = ne

    def __str__(self):
        return self.token

    def lower(self):
        return self.token.lower()

    def __eq__(self, o):
        if type(self) == type(o):
            return self.token == o.token and self.pos == o.pos and self.ne == o.ne
        else:
            return self.token == o

class Document:
    def __init__(self, filename, url, title, highlights, sentences):
        self.filename = filename
        self.url = url
        self.title = title
        self.highlights = highlights
        self.sentences = sentences

def read_line_document(line):
    filename, url, title, hl, sents = line.strip().split("\t")

    highlights = list()
    for h in hl.split("|"):
        tokens = list()
        for token in h.split():
            word, pos, ne = token.split("/")
            tokens.append(Token(word, pos, ne))
        highlights.append(tokens)
    sentences = list()
    for s in sents.split("|"):
        tokens = list()
        for token in s.split():
            word, pos, ne = token.split("/")
            tokens.append(Token(word, pos, ne))
        sentences.append(tokens)

    return Document(filename, url, title, highlights, sentences)

def read_vocab(path):
    id2vocab = list()
    vocab2id = dict()
    with open(path, "r") as f:
        for line in f:
            token = line.strip()
            vocab2id[token] = len(id2vocab)
            id2vocab.append(token)
    return id2vocab, vocab2id
