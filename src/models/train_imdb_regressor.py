#!/usr/bin/env python
import argparse
import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

def parse_runtime(s):
    if pd.isna(s): return np.nan
    m = re.search(r'(\d+)', str(s))
    return float(m.group(1)) if m else np.nan

def parse_gross(s):
    if pd.isna(s): return np.nan
    digits = re.sub(r'[^0-9]', '', str(s))
    return float(digits) if digits else np.nan

def preprocess(df):
    df = df.copy()
    df['Runtime_min'] = df['Runtime'].apply(parse_runtime)
    df['Gross_num'] = df['Gross'].apply(parse_gross)
    # Ano como int
    df['Released_Year_num'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    # Simplificar gênero como lista
    df['Genre_list'] = df['Genre'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    # Multi-hot simples: mantém string original e deixa OneHot cuidar
    df['Genre_simplified'] = df['Genre_list'].apply(lambda lst: lst[0] if len(lst)>0 else 'Unknown')
    # Certificado como categoria
    df['Certificate'] = df['Certificate'].fillna('Unknown')
    # Meta_score / Votes
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')
    df['Overview'] = df['Overview'].fillna('')
    return df

def build_pipeline():
    numeric_features = ['Runtime_min','Meta_score','No_of_Votes','Gross_num','Released_Year_num']
    categorical_features = ['Certificate','Genre_simplified']
    text_feature = 'Overview'

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('txt', TfidfVectorizer(max_features=2000, ngram_range=(1,2)), text_feature)
    ], remainder='drop')

    model = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV com os dados (inclui coluna target)')
    ap.add_argument('--target', default='imdb_rating', help='Nome da coluna target (nota IMDb)')
    ap.add_argument('--out', default='models/imdb_regressor.pkl', help='Caminho do .pkl de saída')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' não encontrado no CSV.")
    y = pd.to_numeric(df[args.target], errors='coerce')
    X = df.drop(columns=[args.target])
    Xp = preprocess(X)

    pipe = build_pipeline()
    pipe.fit(Xp, y)

    # Métricas simples in-sample (para sanity check; ideal é validação cruzada)
    y_pred = pipe.predict(Xp)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)

    metrics = {'rmse_in_sample': rmse, 'mae_in_sample': mae, 'r2_in_sample': r2}
    with open(os.path.join(os.path.dirname(args.out), 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
