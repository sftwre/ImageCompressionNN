import pandas as pd
import wget
import argparse
import multiprocessing as mp


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
    parser.add_argument("--num_images", type=int, default=250)

    args = parser.parse_args()

    # create thread pool
    pool = mp.Pool(mp.cpu_count())

    # read image urls from csv
    df = pd.read_csv(args.csv)

    # download num_images for each process
    batch = 0
    num_images = args.num_images
    image_urls = list(df.TIFF)

    for p in range(mp.cpu_count()):
        pool.apply_async(fetch_data, args=(image_urls[batch: batch + num_images], args.save_path))
        batch += num_images


if __name__ == "__main__":
    main()
