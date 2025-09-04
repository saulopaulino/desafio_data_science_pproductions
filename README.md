# 🎬 EDA de IMDb — Projeto Indicium para PProductions

<p align="center">
  <img src="images/logo_eda_imdb.png" alt="Logo do Projeto" width="250"/>
</p>

## 📌 Contexto

Este repositório foi desenvolvido como parte de um projeto da **Indicium** para o estúdio de Hollywood **PProductions**.  
O objetivo é **analisar o banco de dados do IMDb** e propor **estratégias de desenvolvimento de filmes** que maximizem receita (ROI) e sucesso de crítica/público.

> **Autor:** Saulo Paulino  
> Graduando em Ciências de Dados com Ênfase em Inteligência Artificial (5º período) — UniAmérica  
> Projeto de portfólio para entrevistas e cases técnicos.

---

## 📊 Sumário Executivo

- **Objetivo:** orientar qual tipo de filme deve ser produzido pelo estúdio.  
- **Base de dados:** IMDb (Top 1.000 filmes) + modelagem preditiva + benchmarks de mercado.  
- **Recomendação principal:** Thriller/Horror “elevado” (alto conceito, médio orçamento, 95–110 min, lançamento set/out).  
- **Alternativa:** Aventura/Comédia familiar live-action (PG), com apelo nostálgico e alto potencial internacional.  

📑 [Apresentação completa em PPTX](reports/Apresentacao_PProductions_Indicium_SauloPaulino.pptx)

---

## 🛠️ Metodologia

1. **Preparação de dados**
   - Limpeza de colunas (Gross, Runtime, Year, Gênero primário).
   - Criação de variáveis derivadas (log(Gross), duração em minutos).

2. **EDA (Análise Exploratória)**
   - Distribuições (IMDb rating, Gross).  
   - Tendência temporal (receita média por ano).  
   - Correlações (Votos × Receita, Meta_score × IMDb).  
   - Receita mediana por gênero.  

3. **Modelagem**
   - **Random Forest Regressor** para:
     - Receita (log(Gross))  
     - Nota IMDb  
   - **Métricas (hold-out 25%):**
     - Receita → RMSE ≈ 1,55 | R² ≈ 0,51  
     - IMDb Rating → RMSE ≈ 0,21 | R² ≈ 0,38  
   - Importância das variáveis:
     - **No_of_Votes** (popularidade) foi o maior driver de receita.  
     - **Ano, Runtime e Meta_score** também tiveram peso relevante.  
     - Certificado (PG/PG-13) e gênero modulam alcance.

---

## 📈 Resultados Principais

- 🎥 **Votos no IMDb** explicam grande parte da receita: filmes populares convertem mais em bilheteria.  
- 📅 **Ano**: títulos mais recentes tendem a gerar maior receita.  
- ⭐ **Meta_score** influencia positivamente a nota do IMDb e tem correlação moderada com receita.  
- 🏷️ **Certificado (rating)** e **gênero** modulam público-alvo e ROI esperado.  

---

## 🎯 Recomendação de Produção

### Opção 1 — ROI Ótimo
- Gênero: **Thriller/Horror** de alto-conceito.  
- Orçamento: **US$ 15–25M** (eficiência de capital).  
- Certificado: **PG-13** (ou R leve).  
- Duração: **95–110 min**.  
- Janela: **setembro–outubro** (festivais + Halloween).  
- Casting: 1 nome reconhecível + elenco emergente.

### Opção 2 — Escala Global
- Gênero: **Aventura/Comédia familiar live-action (PG)**.  
- Orçamento: **US$ 60–90M**.  
- Diferenciais: mascote ou “buddy”, merchandising, música original.  
- Janela: **junho (verão)** ou **novembro (feriados)**.  

---

## 📂 Estrutura do Repositório

├── data/ # datasets (IMDb + derivados)
├── notebooks/ # Jupyter Notebooks de EDA e modelagem
├── src/ # scripts reutilizáveis
├── reports/ # relatórios em PDF e apresentação em PPTX
├── images/ # logo e figuras do projeto
└── README.md # este arquivo

yaml
Copiar código

---

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/eda-imdb.git
   cd eda-imdb
