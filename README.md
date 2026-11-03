# Active Learning for Multi-Label Data Streams

This project investigates active learning strategies for
multi-label data streams using Binary Relevance and online
classifiers from CapyMOA.

Implemented strategies:
- Random Sampling
- Uncertainty Sampling
- Query By Committee

Base classifier:
- Adaptive Random Forest

# Active Learning for Multi-Label Data Streams

Aprendizado ativo para classificação online multirrótulo (OMLC), desenvolvido na Iniciação Científica
FAPESP (Proc. 2026/06454-3), ICMC-USP, sob orientação do Prof. Dr. Ricardo Cerri.

> ⚠️ Projeto em andamento: os resultados são parciais e as estratégias seguem em aperfeiçoamento.

## Visão geral
Compara estratégias de consulta sob orçamento de anotação em fluxos multirrótulo, incluindo a
estratégia proposta **Cluster Hardness Sampling (CHS)** e suas combinações com incerteza e QBC.

- **Classificador:** Binary Relevance + Adaptive Random Forest (CapyMOA)
- **Estratégias:** Random, Uncertainty, QBC, CHS (global e per-label), CHS+Uncertainty, CHS+QBC,
  além das referências No Query e Full Supervision
- **Cenários:** online e baseado em pool
- **Avaliação:** test-then-train (prequential); 30% iniciais para treino e 70% como fluxo, sem shuffle
- **Métricas:** Hamming Loss, Exact Match, F1 macro, % de consultas e tempo de execução

## Bases de dados
Flags, Emotions, Scene, Birds, Yeast, Water-quality (reais), SynHPGrad e SynHPInc (sintéticas).
Fonte e instruções de download: [link/pasta `data/`].

## Estrutura do repositório
```
[pasta/]   descrição
```

## Instalação
```bash
git clone https://github.com/iarabertopena/active-learning-for-multi-label-streams
cd active-learning-for-multi-label-streams
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Versão do Python: [x.x]. Versão do CapyMOA: [x.x].

## Como executar
```bash
python [script.py] --dataset [nome] --strategy [chs_global] --budget 0.3 --scenario [online|pool]
```
Parâmetros principais: [estratégia, orçamento, nº de clusters, pool size, semente].

## Resultados
Os resultados ficam em `[pasta/]` (formato [csv]). Para reproduzir tabelas e figuras: `[comando]`.

## Como citar / Contato
Iara Berto Pena — [e-mail]. Licença: [MIT/outra].
