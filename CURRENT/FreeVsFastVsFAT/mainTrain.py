import os
import argparse
from pyfiglet import figlet_format

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FreeVsFast main Training')
    parser.add_argument('-attack', default='free', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    a = figlet_format(args.attack.upper(), font='starwars')
    print(a)
    os.system("python3 " + args.attack + ".py")

if __name__ == '__main__':
    main()