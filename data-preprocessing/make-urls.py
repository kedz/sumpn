import os

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

train_path = os.path.join("neuralsum", "dailymail", "training")
dev_path = os.path.join("neuralsum", "dailymail", "validation")
test_path = os.path.join("neuralsum", "dailymail", "test")


print "Writing training urls..."
write_urls(train_path, "data/dailymail.urls.train.tsv")
print "Writing validation urls..."
write_urls(dev_path, "data/dailymail.urls.dev.tsv")
print "Writing test urls..."
write_urls(test_path, "data/dailymail.urls.test.tsv")


