import pandas as pd
import wget
import argparse
import multiprocessing as mp

batch = 0

def fetch_data(csv, save_path, num_images):
    """
    Reads csv file into pandas data frame and downloads each image
    from the url specified in the TIFF column.
    :return:
    """

    df = pd.read_csv(csv)

    global batch

    image_urls = list(df['TIFF'])

    if batch < len(image_urls):
        urls = image_urls[batch: batch + num_images]

    else:
        return

    batch += num_images

    # download images
    for url in urls: wget.download(url, out=save_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=250)

    args = parser.parse_args()

    # create thread pool
    pool = mp.Pool(mp.cpu_count())

    # read data from csv and download images
    pool.apply(fetch_data, args=(args.csv, args.save_path, args.chunk_size))

if __name__ == "__main__":
    main()
