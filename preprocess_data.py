# -*- coding: utf-8 -*-



import argparse
from audiomnist.io.preprocess_data import preprocess_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-src', default=os.path.join(os.getcwd(), "data"), help="Path to folder containing each participant's data directory.")
    parser.add_argument('--destination', '-dst', default=os.path.join(os.getcwd(), "preprocessed_data"), help="Destination where preprocessed data shall be stored.")
    parser.add_argument('--meta', '-m', default=os.path.join(os.getcwd(), "data", "audioMNIST_meta.txt"), help="Path to meta_information json file.")

    args = parser.parse_args()

    # preprocessing
    preprocess_data(src=args.source, dst=args.destination, src_meta=args.meta)