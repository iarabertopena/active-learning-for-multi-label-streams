# Active Learning for Multi-Label Data Streams

Aprendizado ativo para classificação online multirrótulo (OMLC), desenvolvido na Iniciação Científica
FAPESP (Proc. 2026/06454-3), ICMC-USP, sob orientação do Prof. Dr. Ricardo Cerri.

> Projeto em andamento: os resultados são parciais e as estratégias seguem em aperfeiçoamento.

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
