# PProductions — Projeto IMDb (Indicium)

## Sumário Executivo
Baseado no arquivo `desafio_indicium_imdb.csv` (anexo), este repositório contém: EDA, recomendação não‑personalizada, identificação de fatores de receita e um modelo de **regressão** para prever **nota IMDb** (salvo em `models/imdb_regressor.pkl`).

---

## Principais Achados
- **Tamanho do dataset**: 999 linhas, 23 colunas.
- **Gêneros mais frequentes**: {'Drama': 288, 'Action': 172, 'Comedy': 155, 'Crime': 107, 'Biography': 88, 'Animation': 82, 'Adventure': 72, 'Mystery': 12, 'Horror': 11, 'Western': 4}.
- **Recomendação (pessoa desconhecida)** — fórmula IMDb com **m=461994** e **C=7,948**: **The Godfather** (WR=8,922, IMDb=9,200, votos=1620367).
- **Modelo IMDb (holdout 20%)** — RMSE=0,207, MAE=0,168, R²=0,350.
- **NLP (Overview → Gênero)** — Estimativa qualitativa: texto agrega sinal; métrica quantitativa não calculada nesta execução..

### Fatores de Receita (`Gross`)
Modelo de `RandomForest` sobre `log(Gross+1)` com numéricos + categóricos (sem texto). R² teste=0,545.  
Top *features* (até 10): [{'feature': 'No_of_Votes', 'importance': 0.4887084998239489}, {'feature': 'Released_Year_num', 'importance': 0.1425247727777743}, {'feature': 'Meta_score', 'importance': 0.09583119589874291}, {'feature': 'Runtime_min', 'importance': 0.0798422252779035}, {'feature': 'Certificate_Unknown', 'importance': 0.05417821666976803}, {'feature': 'Certificate_R', 'importance': 0.035243011150818765}, {'feature': 'Genre_simplified_Animation', 'importance': 0.011847232028242036}, {'feature': 'Genre_simplified_Crime', 'importance': 0.011165883343717199}, {'feature': 'Genre_simplified_Drama', 'importance': 0.009586762728250216}, {'feature': 'Genre_simplified_Action', 'importance': 0.009520955955170423}].  
> Observação: recomendamos enriquecer com orçamento, janela de lançamento (mês/feriados) e premiações para maior poder explicativo.

---

## Respostas Solicitadas

**Qual filme recomendar para uma pessoa que você não conhece?**  
→ **The Godfather**. Critério: *weighted rating* balanceando qualidade e popularidade. Veja `reports/final/top_recommendations.csv` para o top‑5.

**Principais fatores ligados a alta expectativa de faturamento?**  
→ Popularidade/engajamento (**No_of_Votes**), qualidade crítica (**Meta_score**), **Gênero**, certificação (**Certificate**) e temporalidade (**Released_Year_num**) aparecem como determinantes no modelo; efeitos são não lineares.

**Insights da coluna `Overview`. É possível inferir o gênero?**  
→ Sim, qualitativamente o texto agrega sinal (palavras/expressões típicas por gênero). Recomenda-se combiná-lo com metadados; a métrica quantitativa será adicionada quando apropriado.

**Como prever a nota do IMDb?**  
- **Problema**: regressão.  
- **Variáveis**: numéricos (`Runtime_min`, `Meta_score`, `No_of_Votes`, `Gross_num`, `Released_Year_num`), categóricos (`Certificate`, `Genre_simplified`) e texto (`Overview` via TF‑IDF).  
- **Modelo**: `GradientBoostingRegressor` em `Pipeline` com `ColumnTransformer`.  
- **Métricas**: RMSE/MAE (holdout 20%).  
- **Prós/Contras**: bom para não linearidades; menor interpretabilidade que regressão linear; pode exigir *tuning*.

**Nota IMDb para o filme com as características fornecidas (Shawshank)**  
→ **8,729** (0–10).

**Modelo salvo (.pkl)**  
→ `models/imdb_regressor.pkl` (treinado a partir de `data/processed/movies.csv`).

---

## Como Rodar

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Treinar
python src/models/train_imdb_regressor.py --input data/processed/movies.csv --target imdb_rating --out models/imdb_regressor.pkl

# Prever exemplo
python src/models/predict_single.py --model models/imdb_regressor.pkl --json examples/shawshank.json
```

## Estrutura
```
data/processed/movies.csv
models/imdb_regressor.pkl
reports/final/top_recommendations.csv
src/models/train_imdb_regressor.py
src/models/predict_single.py
examples/shawshank.json
README.md
```
