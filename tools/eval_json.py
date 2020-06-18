import os
import sys
import argparse

sys.path.insert(0, '../lib')
from utils import misc_utils
from evaluate import compute_JI, compute_APMR

def eval_all(args):
    # ground truth file
    gt_path = '/data/CrowdHuman/annotation_val.odgt'
    assert os.path.exists(gt_path), "Wrong ground truth path!"
    misc_utils.ensure_dir('outputs')
    # output file
    eval_path = os.path.join('outputs', 'result_eval.md')
    eval_fid = open(eval_path,'a')
    eval_fid.write((args.json_file+'\n'))
    # eval JI
    res_line, JI = compute_JI.evaluation_all(args.json_file, 'box')
    for line in res_line:
        eval_fid.write(line+'\n')
    # eval AP, MR
    AP, MR = compute_APMR.compute_APMR(args.json_file, gt_path, 'box')
    line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
    print(line)
    eval_fid.write(line+'\n\n')
    eval_fid.close()

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-f', default=None, required=True, type=str)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_eval()
