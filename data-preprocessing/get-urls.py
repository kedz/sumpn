import os
from subprocess import check_output
import json

key = "53NLTBGaWE9NOgjsHlnE7L0viQHv8BTQUd44LmH6"

path = "../data/dailymail.urls.dev.tsv"
output = "../data/dailymail.content.dev.tsv"


args = ['curl', '-s', '-H', '"x-api-key: {}"'.format(key)]
url_template = '"https://mercury.postlight.com/parser?url={}"'

lines = 1

bad_paths = []

with open(path, "r") as f, open(output, "w") as o:
    for line in f:
        filename, url = line.strip().split()
        command = " ".join(args + [url_template.format(url)])
        res = check_output(command, shell=True)
        try:
            obj = json.loads(res)
            o.write(filename)
            o.write("\t")
            o.write(url)
            o.write("\t")
            o.write(obj["title"])
            o.write("\t")
            cont = obj["content"].replace(
                "\n", "__NL__").replace("\t", "__TAB__")
            o.write(cont)
            o.write("\n")
        except:
            bad_paths.append(path)
        lines += 1

print "bad paths"
for path in bad_paths:
    print path
