# Multiplicação de Matrizes Otimizada em C

## 📌 Descrição do Projeto
Este projeto tem como objetivo implementar e analisar o desempenho da **multiplicação de matrizes** em linguagem **C**, comparando uma versão **não otimizada** com uma versão **otimizada**, utilizando técnicas de **paralelização com OpenMP** e **análise de desempenho com PAPI**.

O foco principal é avaliar o impacto das otimizações no tempo de execução e no uso da hierarquia de memória, especialmente em matrizes de grande porte.

---

## 🎯 Objetivos
- Implementar a multiplicação de matrizes em C
- Aplicar otimizações de desempenho
- Utilizar paralelização com **OpenMP**
- Coletar métricas de hardware com **PAPI**
- Comparar desempenho entre versões otimizada e não otimizada
- Analisar o impacto de cache (L1 e L2)

---

## 🛠️ Tecnologias Utilizadas
- Linguagem **C**
- **OpenMP** (paralelização)
- **PAPI** (Performance API)
- Sistema Operacional **Linux**
- Compilador **GCC**

---

## 📊 Métricas Analisadas
Para evitar ruído e imprecisão nos resultados, foram selecionadas métricas específicas:

- **L1_DCM** – L1 Data Cache Misses
- **L2_DCM** – L2 Data Cache Misses
- Tempo de execução
- Comparação entre versões

> Observação: o uso excessivo de métricas pode gerar análises imprecisas, por isso o projeto foca em métricas relevantes à hierarquia de memória.

---

## 📐 Tamanho das Matrizes
Os testes foram realizados com matrizes quadradas de diferentes tamanhos, incluindo:
- 1000 x 1000
- 2000 x 2000
- Até 4000 x 4000

---

## ▶️ Como Compilar
Certifique-se de ter o **GCC**, **OpenMP** e **PAPI** instalados.

Exemplo de compilação:

gcc -fopenmp main.c matrix.c -lpapi -o matrix_mult

---

## 👤 Autor

- Samuel Villela
- Colaboradores: Alexandre Blandino e Murilo Caetano
- Estudantes de Ciência da Computação
- Interesse em otimização, paralelismo e análise de desempenho


