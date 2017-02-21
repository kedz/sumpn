from __future__ import print_function
import os
import sys
import argparse

def main():

    help_message = "Split files into chunks."

    parser = argparse.ArgumentParser(help_message)
    parser.add_argument('--input', required=True,
        help="Location of content tsv file.")
    parser.add_argument('--output', required=True,
        help="Location to write output data.")
    parser.add_argument('--size', type=int, required=True, 
        help="Location to write output data.")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    filename = os.path.basename(args.input)

    if args.size < 1:
        print("Chunk size must be a positive integer.")
        sys.exit()
    
    file_num = 1
    current_size = 0
    
    output_path = os.path.join(args.output, "{}-{}".format(filename, file_num))
    o = open(output_path, "w")
    
    print("Writing to file {} ...".format(output_path))
    with open(args.input, "r") as f:
        
        for line in f:
            if current_size == args.size:
                o.close()
                file_num += 1
                current_size = 0
                output_path = os.path.join(
                    args.output, "{}-{}".format(filename, file_num))
                o = open(output_path, "w")
                print("Writing to file {} ...".format(output_path))

            o.write(line)
            current_size += 1
    o.close() 
            
if __name__ == "__main__":
    main()
