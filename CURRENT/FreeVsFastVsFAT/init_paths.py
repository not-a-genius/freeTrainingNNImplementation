import sys
import matplotlib
import os

matplotlib.use('Agg')
sys.path.insert(0, 'lib')

try:
    os.makedirs('trained_models')
except Exception as e:
    pass