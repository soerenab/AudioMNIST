import argparse
from audiomnist.train import alexnet

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Testing script for tensorflow.keras AlexNet model.")
    parser.add_argument('-i','--input_dataset', help="path to TFRecord file", required=True)
    parser.add_argument('-o','--checkpoint_output', help="path to checkpoint folder", required=True)
    parser.add_argument('-e','--epoch', help="epoch to test", required=True, type=int)
    parser.add_argument('-b','--batch_size', help="Batch size", required=True, type=int)


    args = parser.parse_args()

    audionet.test(args.input_dataset, args.checkpoint_output, args.epoch, args.batch_size)