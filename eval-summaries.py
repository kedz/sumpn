import os
from itertools import izip
from tempfile import NamedTemporaryFile
import subprocess
import pandas as pd
pd.set_option('display.width', 1000)

def read_paths(path):

    paths = [os.path.join(path, filename) for filename in os.listdir(path)]
    paths.sort()
    return paths

def check_parity(paths1, paths2):
    if len(paths1) != len(paths2):
        raise Exception("Size of summary lists not equal!")
    for p1, p2 in izip(paths1, paths2):
        f1 = os.path.split(p1)[-1]
        f2 = os.path.split(p2)[-1]
        if f1 != f2:
            raise Exception(f1 + " not equal to " + f2 + " --- check inputs!")

def evaluate(system_paths, reference_paths, rouge_path):

    check_parity(system_paths, reference_paths)

    template = r"perl {}/ROUGE-1.5.5.pl -e {}/data -n 2 -s -b {} -z SPL {}"

    with NamedTemporaryFile("w", delete=False) as f:
        for p1, p2 in izip(system_paths, reference_paths):
            f.write("{} {}\n".format(os.path.abspath(p1), os.path.abspath(p2)))

    data = list()
    
    for size in [75, 250]:
        command = template.format(rouge_path, rouge_path, size, f.name)
        output = subprocess.check_output(command, shell=True)

        for line in output.split("\n"):
            if line.startswith("X ROUGE-1 Average_R:"):
                data.append(float(line.split()[3]))
            elif line.startswith("X ROUGE-2 Average_R:"):
                data.append(float(line.split()[3]))
            elif line.startswith("X ROUGE-L Average_R:"):
                data.append(float(line.split()[3]))

    os.remove(f.name)
    columns = pd.MultiIndex.from_tuples(
        [('75 bytes','1'), ('75 bytes','2'), ('75 bytes', 'L'),
         ('250 bytes','1'), ('250 bytes','2'), ('250 bytes', 'L')],
        names=['length', 'order'])
    return pd.DataFrame([data], columns=columns)

def main():

    import argparse

    hlp = "Evaluate summaries with ROUGE."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--reference', required=True,
        help="Reference summary directory")
    parser.add_argument('--systems', required=True, nargs='+',
        help="System summary directories")
    parser.add_argument('--rouge', required=True,
        help="ROUGE root directory")
    
    args = parser.parse_args()
    
    print "reading from " + args.reference + " ..."
    ref_paths = read_paths(args.reference)

    dfs = list()
    for system in args.systems:
        print "reading from " + system + " ..."
        system_paths = read_paths(system)
        df = evaluate(system_paths, ref_paths, os.path.abspath(args.rouge))
        dfs.append(df)

    df = pd.concat(dfs) 
    df.index = args.systems
    print df

if __name__ == "__main__":
    main()
