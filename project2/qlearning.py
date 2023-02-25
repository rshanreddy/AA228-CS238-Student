import pandas as pd 
import numpy as np
import sys

S, A, gamma, alpha = None, None, None, None

def main():
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    
    global S, A, discount, alpha
    if len(sys.argv) != 4:
        print("Usage should be: qlearning.py <file_type> <alpha> <k_max>")

    if inputfilename == 'small':
        outputfilename = 'small.policy'
        S = 100 # number of states
        A = 4 # number of possible actions
        discount = 0.95
    elif inputfilename == 'medium':
        outputfilename = 'medium.policy'
        S = 50000 
        A = 7
        discount = 1
    elif inputfilename == 'large':
        outputfilename = 'large.policy'
        S = 312020
        A = 9
        discount = 0.95
    else:
        print("No specified file type")

    alpha = float(sys.argv[2])
    k_max = int(sys.argv[3])
    compute(inputfilename, outputfilename, k_max)

def compute(infile, outfile, k_max):
    sars_ = pd.read_csv(infile)
    Q = np.zeros((S,A))
    for k in range(k_max):
        for idx, row in sars_.iterrows():
            s, a, r, s_ = row['s']-1, row['a']-1, row['r']-1, row['sp']-1
            Q = update_model(Q, s, a, r, s_)
    pi = np.argmax(Q, axis=-1)
    with open(outfile, "w") as f:
        for p in pi:
            f.write(str(p+1)+"\n")



if __name__ == '__main__':
    main()