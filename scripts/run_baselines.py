"""Baseline ML chạy độc lập, không cần hoàn thiện TODO."""
import argparse
from pathlib import Path
from shared.baselines import run_baselines
from shared.data import describe, load_data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',choices=['cnn','nlp'],required=True)
    p.add_argument('--data',default='data')
    p.add_argument('--out',default='runs/my_project')
    args=p.parse_args()
    data=load_data(args.project,args.data)
    print(describe(data))
    run_baselines(data,Path(args.out)/args.project/'baselines')
