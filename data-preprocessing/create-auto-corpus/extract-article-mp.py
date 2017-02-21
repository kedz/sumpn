from __future__ import division
import sys
import os
import argparse

from multiprocessing import Pool, current_process
from subprocess import check_output, CalledProcessError
import json

keys = ["GEklFj2zfoHBoTOxCpXELToglTIq5HeQgvOsPljE",
        "53NLTBGaWE9NOgjsHlnE7L0viQHv8BTQUd44LmH6",
        "c0P9ewsglkr9hBTKCMNyP6NvU3YYjDXH4AZktoTy",
        "XLOWWaTz2jEx6HBwtTFrbHHWbEbsVRjUGfNQNFVA"]

ct = 'curl -s -H "x-api-key: {}" "https://mercury.postlight.com/parser?url={}"'


def process(args):
    filename, url = args
    id = current_process()._identity[0] - 1
    key = keys[id]

    command = ct.format(key, url)
    check_output('curl -s "{}"'.format(url), shell=True)

    try:
        res = check_output(command, shell=True)
        obj = json.loads(res)
        title = obj["title"].encode("utf-8").replace(
            "\n", "__NL__").replace("\t", "__TAB__")
        cont = obj["content"].encode("utf-8").replace(
            "\n", "__NL__").replace("\t", "__TAB__")

        return True, "{}\t{}\t{}\t{}\n".format(filename, url, title, cont)

    except (TypeError, ValueError, KeyError, CalledProcessError), e:
        return False, "{}\t{}\n".format(filename, url)

def main():

    hlp = "Download article content for neuralsum dataset."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--input', required=True,
        help="Location of url tsv file.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
    parser.add_argument('--errors', required=True,
        help="Location to write failed extractions.")
 
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    errors_dir = os.path.dirname(args.errors)
    if errors_dir != '' and not os.path.exists(errors_dir):
        os.makedirs(errors_dir)

    data = list()
    with open(args.input, "r") as f:
        for line in f:
            filename, url = line.strip().split()
            data.append((filename, url))
    n_data = len(data)

    pool = Pool(4)

    with open(args.output, "w") as f, open(args.errors, "w") as ef:
        for i, res in enumerate(pool.imap_unordered(process, data), 1):
            sys.stdout.write("Extracted {:8.4f}% \r".format(100 * i / n_data))
            sys.stdout.flush()
            if res[0]:
                f.write(res[1])
            else:
                ef.write(res[1])

if __name__ == "__main__":
    main()
