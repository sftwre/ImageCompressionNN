import pandas as pd
import wget
import argparse
import multiprocessing as mp

def download_kodak():

    url = "http://r0k.us/graphics/kodak/thumbs/kodim01t.jpg"
    pass


def fetch_data(urls):
   """
   :param urls: list of urls to images 
   :param save_path: path to place downloaded images
   :param num_images:
   :return: 
   """
   # download images
   for url in urls: wget.download(url, out=save_path)



parser = argparse.ArgumentParser()

parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

args = parser.parse_args()

save_path = args.save_path

# create thread pool
num_processes = 10
dataset_size = 1000

pool = mp.Pool(processes=num_processes)

# read image urls from csv
df = pd.read_csv(args.csv)

# compute number of images each process should download
chunksize = dataset_size / num_processes

pool.map(fetch_data, list(df.TIFF), chunksize=chunksize)
