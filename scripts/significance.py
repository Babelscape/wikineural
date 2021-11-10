from argparse import ArgumentParser

import numpy as np
from scipy.stats import ttest_rel


def read_metrics(filename):
    macro_f1 = []
    span_f1 = []

    with open(filename) as f:
        for line in f:
            macro, span = map(float, line.rstrip().split('\t'))
            macro_f1.append(macro)
            span_f1.append(span)

    return macro_f1, span_f1


def mean_std(array):
    return np.mean(array), np.std(array)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('--lang', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--compare', default=None)
    parser.add_argument('-p', default=0.05, type=float)

    return parser.parse_args()


def fmt(array):
    return ' ± '.join(map(lambda e: f"{e:.2f}", mean_std(array)))


def test_significance(key, group1, group2, p):
    statistic, p_val = ttest_rel(group1, group2)

    prefix = ""
    if p_val > p:
        prefix = "NOT "
    print(f"{key}: {prefix}statistically significant at p<={p} (p-value={p_val}, stat={statistic})")


def main(args):

    filename_format = f"results/{{name}}-{args.lang}-{args.dataset}.tsv"

    macro_f1s, span_f1s = read_metrics(filename_format.format(name=args.name))
    print(f"{args.name}: {fmt(macro_f1s)} / {fmt(span_f1s)}")

    if args.compare:
        macro_f1s_comp, span_f1s_comp = read_metrics(filename_format.format(name=args.compare))
        print(f"{args.compare}: {fmt(macro_f1s_comp)} / {fmt(span_f1s_comp)}")
        test_significance('macro', macro_f1s, macro_f1s_comp, args.p)
        test_significance('span', span_f1s, span_f1s_comp, args.p)


if __name__ == '__main__':
    main(parse_args())
