import pickle
import numpy as np
import sys

def main(fn):
	x = pickle.load(open(fn, 'rb'))
	s = x['epiosde_scores'][0]
	print(len(s), np.mean(s[-5000:]), np.mean(s[-2000:]), np.mean(s[-1000:]))

if __name__ == "__main__":
	main(sys.argv[1])
