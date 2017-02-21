import os
import argparse

def get_url(path):
    with open(path, "r") as f:
        url = f.readline().strip()
    return url

def write_urls(input_dir_path, output_path):

    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [os.path.join(input_dir_path, file) 
             for file in os.listdir(input_dir_path)]
    files.sort()
    with open(output_path, "w") as f:
        for file in files:
            filename = os.path.basename(file)
            url = get_url(file)
            f.write("{}\t{}\n".format(filename, url))

def main():

    hlp = "Extract urls from neuralsum dataset."

    parser = argparse.ArgumentParser(hlp)
    parser.add_argument('--input-dir', required=True,
        help="Location of data split directory.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
 
    args = parser.parse_args()
    write_urls(args.input_dir, args.output)

if __name__ == "__main__":
    main()
