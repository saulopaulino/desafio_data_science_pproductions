# PProductions — Análise Cinematográfica orientada a Receita

> **Contexto narrativo**  
Você foi alocado em um time da **Indicium** contratado por um estúdio de Hollywood chamado **PProductions**. Sua missão é analisar um banco de dados cinematográfico para orientar **qual tipo de filme** deve ser o próximo a ser desenvolvido. Há muito dinheiro envolvido, então a análise deve ser **detalhada** e considerar o máximo de fatores possível. A **introdução de dados externos é permitida e encorajada**.

---

## Objetivos do projeto

1. **EDA completa**: entender a distribuição das variáveis, relações entre elas e levantar hipóteses.  
2. **Recomendar um filme para uma pessoa desconhecida** (cold-start).  
3. **Identificar fatores ligados a alta expectativa de faturamento** (proxy: `Gross`/receita).  
4. **Extrair insights a partir de `Overview`** (NLP): inferir **gênero** a partir do texto é viável?  
5. **Prever nota do IMDb**: formular o problema, escolher variáveis/transformações, selecionar modelo, justificar prós/cons e **métrica**.  
6. **Estimar a nota IMDb** do filme com características fornecidas (ex.: *The Shawshank Redemption*).  
7. **Entregar artefatos reprodutíveis**: relatórios, código, e **modelo salvo em `.pkl`**.

---

## Dados

- **Internos**: dataset principal (ex.: `data/processed/movies.csv`) contendo colunas como:  
  `Series_Title, Released_Year, Certificate, Runtime, Genre, Overview, Meta_score, Director, Star1..Star4, No_of_Votes, Gross, imdb_rating (target)`.

- **Externos (opcionais)**: sugeridos para enriquecer o modelo/EDA:  
  - Tendências de busca (**Google Trends**) por franquias/atores/diretores.  
  - Orçamento (`Budget`), países/idiomas de produção (TMDb/OMDb).  
  - Calendário de feriados/estação de lançamento (efeitos sazonais em bilheteria).  
  - Premiações (Oscars, Globes) e indicações por ano/gênero.  
  - Concorrência no período (quantidade de lançamentos no mês).

> **Observação**: os dados originais não são versionados no Git. Coloque-os localmente em `data/raw/` ou `data/processed/`.

---

## Entregas

- **Relatório de EDA** (notebook ou PDF) em `reports/final/` (e/ou `notebooks/EDA.ipynb`).  
- **Código de modelagem** (pode estar no notebook, mas scripts em `src/models/` são bem-vindos).  
- **Modelo salvo `.pkl`** em `models/`.  
- **README** (este arquivo) com instruções de instalação e execução.  
- **Arquivo de requisitos** com versões exatas (`requirements-freeze.txt`).

---

## Como rodar

```bash
# 1) criar e ativar ambiente virtual (exemplo com venv)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) instalar dependências
pip install -r requirements.txt
# opcionalmente: pip install -r requirements-freeze.txt

# 3) preparar dados
# coloque seu CSV tratado em: data/processed/movies.csv

# 4) treinar o modelo de previsão de IMDb
python src/models/train_imdb_regressor.py --input data/processed/movies.csv --target imdb_rating --out models/imdb_regressor.pkl

# 5) prever nota para um exemplo
python src/models/predict_single.py --model models/imdb_regressor.pkl --json examples/shawshank.json
```

Os artefatos (métricas e o `.pkl`) serão gravados em `models/`.

---

## Metodologia sugerida

### 1) EDA
- Consistência: tipos corretos (anos, números, categorias), tratamento de *outliers* (`Gross`, `No_of_Votes`), normalização de `Runtime` (minutos).  
- Correlações e relações não-lineares (ex.: `No_of_Votes` vs. `imdb_rating`).  
- Séries temporais: tendências por `Released_Year`.  
- **Texto**: nuvem de palavras/TF‑IDF para `Overview`, tópicos (LDA), sentimento.  
- **Hipóteses** (exemplos):  
  - H1: dramas biográficos com alto *Meta_score* e elencos premiados têm maior **imdb_rating**.  
  - H2: ação/aventura com lançamentos em feriados de verão/outono elevam **Gross**.  
  - H3: *Overview* com termos ligados a “esperança/redemption/friendship” está positivamente associada a notas altas.

### 2) Recomendar para uma pessoa desconhecida
- Critério **popularidade x qualidade**: recomendação *non-personalized* baseada em **imdb_rating** ajustada por **No_of_Votes** (para evitar viés de poucos votos) e *freshness* (ano).

### 3) Fatores ligados a alta receita
- Regressão ou *uplift* sobre `Gross` (ou log‑`Gross`) com preditores: `Genre` (one‑hot/multi‑label), `Runtime`, sazonalidade (mês/feriados), `Meta_score`, estrelas/diretores (hashing ou *target encoding*), *franchise flag*, *sequel flag*, orçamento (se disponível) e *marketing proxies* (Trends).

### 4) Insights da coluna `Overview` e inferência de gênero
- **NLP**: TF‑IDF (unigram/bigram), normalização (lowercase, remoção de *stopwords*), stemming/lemmatização.  
- **Classificação de gênero** (multi-label ou multi‑classe simplificada). Métricas: F1 macro/micro, ROC‑AUC.  
- **Interpretação**: pesos de termos, SHAP para textos.

### 5) Previsão de `imdb_rating`
- **Problema**: **regressão**.  
- **Candidatos**: Linear Ridge/Lasso, RandomForest, Gradient Boosting, XGBoost/LightGBM, CatBoost.  
- **Pipeline** (exemplo implementado): `ColumnTransformer` com:  
  - numéricos: `Runtime_min`, `Meta_score`, `No_of_Votes`, `Gross_num` (log1p) → *scaler*;  
  - categóricos: `Certificate`, `Genre` (multi-hot simplificado via split de vírgulas), ano (`Released_Year` binned);  
  - texto: `Overview` → `TfidfVectorizer(max_features=2000, ngram_range=(1,2))`.  
- **Modelo**: `GradientBoostingRegressor` (robusto a não‑linearidades, bom *baseline*).  
- **Métrica**: **RMSE** (erro médio quadrático raiz) + **MAE** para interpretação.

### 6) Estimar IMDb de *The Shawshank Redemption*
Após treinar o pipeline, rode `predict_single.py` com o JSON de exemplo para obter a predição — o script imprime a nota estimada (0–10).

---

## Estrutura do repositório

```
.
├── data/
├── notebooks/
├── src/
├── reports/
├── models/
├── requirements.txt
├── requirements-freeze.txt
├── .gitignore
└── README.md
```

---

## Qualidade e Boas Práticas
- Organização em módulos (`src/`) e *pipelines* reprodutíveis.  
- *Type hints*, *logging*, docstrings, pep8/ruff (opcional).  
- Testes unitários básicos em `tests/`.  
- Scripts e notebooks com seções claras e *seed* fixo.

---

*Gerado em September 04, 2025.*
