import os
import argparse
from pyfiglet import figlet_format

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FreeVsFast main Training')
    parser.add_argument('--adv-learning', default='free', type=str,choices=['free', 'fast'],
        help="Choose adv. learning algorithm between: free, fast ")
    return parser.parse_args()

def main():
    args = parse_args()
    a = figlet_format(args.adv_learning.upper(), font='starwars')
    print(a)
    os.system("python3 " + args.adv_learning + ".py")

if __name__ == '__main__':
    main()