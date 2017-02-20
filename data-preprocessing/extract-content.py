from __future__ import print_function, division
import os
import sys
import argparse

import numpy as np
from HTMLParser import HTMLParser
import re
from unidecode import unidecode

pattern = r'<p class="mol-para-with-font"><font>(.*?)</font>'
list_body_patt = r'<ul class="mol-bullets-with-font">(.*?)</ul>'
list_el_patt = r'<li[^>]*?>(.*?)</li>'


#lis<ul class="article-summary news"><li><span>
list_body_patt2 = r"<ul>(.*)</ul>"
list_body_patt3 = r'<ul class="article-summary[^>]*?">(.*?)</ul>'
list_el_patt2 = r"<li[^>]*?>(.*?)</li>"
#<ul><li><span>

input_patt2 = r"<p><font>(.*?)</font>(<br>)?</p>"

h = HTMLParser()
a_patt = r"<a[^>]*>(.*?)</a>"
html_patt = r"<[^>]*?>"

bad_filenames = set([
    "003c80858d89ec2dcf226973061e9e24a4dbcab6.summary",
    "004f2330b86b71d44b5a809e6eab6c49f90ea928.summary"
    ])

def get_highlights(content):
    list_body = re.search(list_body_patt, content, flags=re.DOTALL)
    if list_body != None:
        list_body = list_body.groups()[0]
        
        highlights = list()
        for match in re.findall(list_el_patt, list_body, flags=re.DOTALL):
            highlight = unidecode(h.unescape(match.decode("utf-8")))
            highlights.append(highlight)
        return highlights
    
    list_body = re.search(list_body_patt3, content, flags=re.DOTALL)
    if list_body != None:
        list_body = list_body.groups()[0]
        highlights = list()
        for match in re.findall(list_el_patt2, list_body, flags=re.DOTALL):
            highlight = unidecode(h.unescape(match.decode("utf-8")))
            highlight = re.sub(a_patt, r"\1", highlight)
            highlight = re.sub(html_patt, r" ", highlight)
            highlights.append(highlight)
        
        return highlights
    list_body = re.search(list_body_patt2, content, flags=re.DOTALL)
    if list_body != None:
        list_body = list_body.groups()[0]
        highlights = list()
        for match in re.findall(list_el_patt2, list_body, flags=re.DOTALL):
            highlight = unidecode(h.unescape(match.decode("utf-8")))
            highlight = re.sub(a_patt, r"\1", highlight)
            highlight = re.sub(html_patt, r" ", highlight)
            highlights.append(highlight)
        
        return highlights

   
    return list()


def get_article_grafs(content):
    grafs = list()
    for match in re.findall(pattern, content, flags=re.DOTALL):
        graf_raw = re.sub(a_patt, r"\1", match).decode("utf-8")
        graf = unidecode(h.unescape(graf_raw))
        grafs.append(graf)
    if len(grafs) == 0:
        for match in re.findall(input_patt2, content, flags=re.DOTALL):
            match = match[0]
            graf_raw = re.sub(a_patt, r"\1", match).decode("utf-8")
            graf = unidecode(h.unescape(graf_raw))
            graf = re.sub(html_patt, r" ", graf)
            grafs.append(graf)

    return grafs

def extract_content(input_path, output_path, error_dir):

    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if error_dir != '' and not os.path.exists(error_dir):
        os.makedirs(error_dir)

    highlight_lengths = list()
    graf_lengths = list()
    highlight_byte_lengths = list()
    graf_byte_lengths = list()

    with open(input_path, "r") as f, open(output_path, "w") as o:
        data = list()
        for line in f:
            try:
                filename, url, title, content = line.strip().split("\t")
            except ValueError, e:
                print(e)
                print(line)
            data.append((filename, url, title, content))

        n_data = len(data)

        for i, (filename, url, title, content) in enumerate(data, 1):
            sys.stdout.write("{:6.3f}%     \r".format(100 * i / n_data))
            sys.stdout.flush()

            if filename in bad_filenames: continue

            # unlikely but better to be safe...
            content = content.replace("|", "----") 
            #content = #content.replace("__NL__", "\n").replace("__TAB__", "\t")
            grafs = get_article_grafs(content) 
            highlights = get_highlights(content)

            highlight_lengths.append(len(highlights))
            highlight_byte_lengths.append(sum([len(h) for h in highlights]))
            graf_lengths.append(len(grafs))
            graf_byte_lengths.append(sum([len(g) for g in grafs]))

            if len(highlights) == 0 or len(grafs) == 0: 
                with open(os.path.join(error_dir, filename), "w") as f:
                    f.write(content.replace(
                        "__NL__", "\n").replace("__TAB__", "\t"))

                continue

            output_line = "{}\t{}\t{}\t{}\t{}\n".format(
                filename, url, title,
                u"|".join(highlights).encode("utf-8"),
                u"|".join(grafs).encode("utf-8"))
            o.write(output_line)
        print("")

        

        graf_lengths = np.array(graf_lengths)
        print(np.mean(graf_lengths))
        print(np.mean(graf_byte_lengths))
        print(np.percentile(graf_lengths, [50, 60, 70, 80, 90, 95, 99]))
        print(np.mean(highlight_lengths))
        print(np.mean(highlight_byte_lengths))
        print(np.percentile(highlight_lengths, [50, 60, 70, 80, 90, 95, 99]))
        print(graf_lengths.shape[0])
        print((graf_lengths == 0).sum())


def main():

    hlp = "Extract urls from neuralsum dataset."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--input', required=True,
        help="Location of content tsv file.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
    parser.add_argument('--error-dir', required=True,
        help="Location to write files that we failed to extract.")

    args = parser.parse_args()
    extract_content(args.input, args.output, args.error_dir)

if __name__ == "__main__":
    main()
