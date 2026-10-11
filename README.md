# 🥣 Detecção de Nível de Comedouro: Vasilha Cheia vs. Vasilha Vazia

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?logo=keras)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Colab-GPU%20T4-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Projeto desenvolvido como parte do **Desafio de Redes Neurais e Transfer Learning**. O objetivo da aplicação é utilizar Visão Computacional para monitorar e classificar o status de um comedouro pet em duas classes: **Vasilha Cheia** e **Vasilha Vazia**, comparando o desempenho de uma **arquitetura base** contra uma **arquitetura otimizada**.

---

## 📌 Sumário
* [Objetivo do Projeto](#-objetivo-do-projeto)
* [Classes Analisadas](#-classes-analisadas)
* [Estrutura do Dataset](#-estrutura-do-dataset)
* [Arquiteturas Testadas](#-arquiteturas-testadas)
* [Comparativo de Desempenho](#-comparativo-de-desempenho)
* [Como Executar no Google Colab](#-como-executar-no-google-colab)
* [Conclusões](#-conclusões)
* [Licença](#-licença)

---

## 🎯 Objetivo do Projeto

Substituir sensores mecânicos convencionais de nível/peso por visão computacional direta sobre o recipiente. O pipeline utiliza técnicas de **Transfer Learning** partindo de uma base convolucional pré-treinada no *ImageNet* (MobileNetV2 / Inception), realizando o fine-tuning das camadas densas superiores para a tomada de decisão binária.

---

## 🏷️ Classes Analisadas

| Classe | Rótulo | Descrição Operacional |
| :--- | :---: | :--- |
| **Vasilha Vazia** | `0` | Recipiente sem alimento; indica necessidade de abastecimento imediato. |
| **Vasilha Cheia** | `1` | Recipiente com ração disponível para o consumo. |

---

## 📂 Estrutura do Dataset

As imagens foram organizadas em diretórios separados para treino (80%) e validação (20%):

```text
dataset_comedouro/
├── train/
│   ├── vasilha_cheia/
│   └── vasilha_vazia/
└── val/
    ├── vasilha_cheia/
    └── vasilha_vazia/

- **Pré-processamento:** Redimensionamento padronizado para 160x160 (ou 299x299 caso use Inception) e normalização dos tensores na escala `[-1, 1]`.
```
---

## 🧠 Arquiteturas Testadas

Conforme a diretriz do desafio, foram construídos e comparados dois modelos:

### 1. Modelo Base (Referência)
* **Base Convolucional:** MobileNetV2 pré-treinada (pesos congelados).
* **Camada de Redução:** `GlobalAveragePooling2D`.
* **Classificador:** Camada única `Dense(1, activation='sigmoid')`.
* **Otimizador:** Adam (`lr=0.001`).

### 2. Modelo Otimizado
* **Base Convolucional:** MobileNetV2 (pesos congelados).
* **Camada de Redução:** `GlobalAveragePooling2D`.
* **Camadas Densas Intermediárias:** `Dense(128, activation='relu')` e `Dense(64, activation='relu')`.
* **Regularização:** `Dropout(0.3)` para mitigar sobreajuste (*overfitting*).
* **Classificador:** `Dense(1, activation='sigmoid')`.
* **Otimizador:** Adam com taxa reduzida (`lr=0.0005`).

---

## 📊 Comparativo de Desempenho

| Métrica | Modelo Base | Modelo Otimizado | Variação / Impacto |
| :--- | :---: | :---: | :---: |
| **Acurácia (Treino)** | 99.76% | 100.00% | +0.24% |
| **Loss (Treino)** | 0.0368 | 0.0032 | -0.0336 (-91.3%) |
| **Acurácia (Validação)** | 100.00% | 100.00% | 0.00% (Estabilidade Máxima) |
| **Loss (Validação)** | 0.0264 | 0.0004 | -0.0260 (-98.5%) |
| **Épocas Treinadas** | 5 | 5 | — |

> **Análise dos Resultados:**

> Ambos os modelos alcançaram **100.00% de acurácia** na separação entre as classes de vasilha cheia e vazia. No entanto, o **Modelo Otimizado** apresentou uma perda de validação (*val_loss*) drasticamente menor (**0.0004** contra **0.0264** da versão base), representando uma redução de aproximadamente **98,5% no erro** e garantindo predições com nível de confiança e assertividade consideravelmente mais elevados.


## 🚀 Como Executar no Google Colab

1. Clone ou faça upload do notebook para o seu [Google Colab](https://colab.research.google.com/).
2. Ative a GPU em **Ambiente de execução > Alterar tipo de ambiente de execução > T4 GPU**.
3. Monte o seu Google Drive onde as pastas `vasilha_cheia` e `vasilha_vazia` estão armazenadas:

```python
from google.colab import drive
drive.mount('/content/drive')

Execute as células sequencialmente para gerar o treinamento, validação e plotagem dos gráficos de comparação.
```

💡 Conclusões
O uso de Transfer Learning permitiu que a rede convergisse com alta precisão mesmo com um volume reduzido de imagens por classe.

A adição de camadas densas intermediárias associadas ao Dropout no modelo otimizado permitiu capturar detalhes contextuais da ração dentro da vasilha, melhorando a estabilidade e a acurácia no conjunto de validação.

📄 Licença
Este projeto está sob a licença MIT.
