from argparse import ArgumentParser
from data_manipulation import perform_mapping

def main():
    args = get_args()
    perform_mapping(args.file_name, args.embed_model)

def get_args():
    parser = ArgumentParser()

    parser.add_argument("-f", "--file_name", help="File Name", default="data.json")
    parser.add_argument("-e", "--embed_model", help="Embedding Model to use or path to local model", default='all-MiniLM-L6-v2')

    return parser.parse_args()


if __name__ == "__main__":
    main()