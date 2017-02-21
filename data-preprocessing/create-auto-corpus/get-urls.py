from __future__ import division, print_function
import os
import sys
import argparse
import time
from subprocess import check_output, CalledProcessError
import json

key = "53NLTBGaWE9NOgjsHlnE7L0viQHv8BTQUd44LmH6"

args = ['curl', '-s', '-H', '"x-api-key: {}"'.format(key)]
url_template = '"https://mercury.postlight.com/parser?url={}"'

def process(path, output):

    data = list()

    with open(path, "r") as f:
        for line in f:
            filename, url = line.strip().split()
            data.append((filename, url))

    n_data = len(data)
    bad_data = list()

    with open(output, "w") as o:
        for i, (filename, url) in enumerate(data, 1):
            sys.stdout.write("{:6.2f}%    \r".format(100 * i / n_data))
            sys.stdout.flush()

            tries = 0
            complete = False
            command = " ".join(args + [url_template.format(url)])

            while not complete and tries < 3:
                if tries > 0:
                    
                    #print("retrying " + url)
                    time.sleep(3)

                res = check_output(command, shell=True)
                try:
                    obj = json.loads(res)
                    title = obj["title"].encode("utf-8")
                    cont = obj["content"].encode("utf-8").replace(
                        "\n", "__NL__").replace("\t", "__TAB__")
                    o.write(filename)
                    o.write("\t")
                    o.write(url)
                    o.write("\t")
                    o.write(title)
                    o.write("\t")
                    o.write(cont)
                    o.write("\n")
                    complete = True
                except TypeError, e:
                    print("\n")

                    print(e)
                    print(obj)
                    #sys.stderr.write("Bad url: {}\n".format(url))
                    bad_data.append((filename, url))
                except ValueError, e:
                    print("\n")
                    print(e)
                    print(obj)
                    sys.stderr.write("Bad url: {}\n".format(url))
                    bad_data.append((filename, url))
                except KeyError, e:
                    print("\n")
                    print(e)
                    print(obj)
                    sys.stderr.write("Bad url: {}\n".format(url))
                    bad_data.append((filename, url))
                except CalledProcessError, e:
                    print("\n")
                    print(e)
                    print(obj)
                    sys.stderr.write("Bad url: {}\n".format(url))
                    bad_data.append((filename, url))
                finally:
                    tries += 1

    with open("bad_data.log", "w") as f:
        for filename, url in bad_data:
            f.write("{}\t{}\n".format(filename, url))

def main():

    hlp = "Download article content for neuralsum dataset."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--input', required=True,
        help="Location of url tsv file.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
 
    args = parser.parse_args()
    process(args.input, args.output)

if __name__ == "__main__":
    main()
