#!/usr/bin/env python
import argparse, json, joblib, pandas as pd
from train_imdb_regressor import preprocess  # reuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--json', required=True, help='Arquivo JSON com um único registro')
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    with open(args.json, 'r') as f:
        record = json.load(f)
    df = pd.DataFrame([record])
    df = preprocess(df)
    yhat = pipe.predict(df)[0]
    print(f"Predição IMDb (0-10): {yhat:.2f}")

if __name__ == '__main__':
    main()
