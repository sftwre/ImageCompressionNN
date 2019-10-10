import pandas as pd
import wget
import argparse

def fetch_data(csv, save_path):
    """
    Reads csv file into pandas data frame and downloads each image
    from the url specified in the TIFF column.
    :return:
    """

    df = pd.read_csv(csv)

    for url in df.TIFF: wget.download(url, out=save_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    # read data from csv and download images
    fetch_data(args.csv, args.save_path)

if __name__ == "__main__":
    main()
