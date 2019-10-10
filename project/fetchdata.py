import pandas as pd
import wget
import argparse
import multiprocessing as mp


dataset_size = 1000

def fetch_data(urls, save_path):
   """
   :param urls: list of urls to images 
   :param save_path: path to place downloaded images
   :param num_images:
   :return: 
   """
   # download images
   for url in urls: wget.download(url, out=save_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    # create thread pool
    num_cpus = mp.cpu_count()

    pool = mp.Pool(num_cpus)

    # read image urls from csv
    df = pd.read_csv(args.csv)

    # download num_images for each process
    batch = 0

    # compute number of images each process should download
    num_images = dataset_size / num_cpus
    image_urls = list(df.TIFF)

    for p in range(mp.cpu_count()):
        pool.apply_async(fetch_data, args=(image_urls[batch: batch + num_images], args.save_path))
        batch += num_images


main()
