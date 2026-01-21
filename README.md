# TOEFL11 BERT

## 1. Project Overview
This project aims to build an Automated Essay Scoring (AES) system for TOEFL-style essays using a BERT-based model. 

## 2. Motivation
- Traditional AES relies on handcrafted linguistic features
- This project explores an end-to-end neural approach using pretrained language models

## 3. Model Architecture

## 4. Dataset
- TOEFL11
- Essays are scored in a continuous scale

## 5. Training Details
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Evaluation Metric: Quadratic Weighted Kappa (QWK)

## 6. Current Progress
- [X] Repository initialized
- [X] Environment setup
- [ ] Data preprocessing
- [ ] BERT baseline implementation

## 7. Future Work
- Add attention mechanism
- Prompt-wise analysis
- Performance comparison with baseline models

## Input Representation Visualization

We visualize the input representation of essays at different stages.

**Fig.1. Raw essay input example**  
![Fig.1 Input Example](figs/fig1_input_example.jpeg)

**Fig.2. POS-tag combined input representation**  
![Fig.2 POS-tagged Input](figs/fig2_pos_tagged_input.jpeg)

**Fig.3. BERT input representation with self-attention (BertViz)**  
![Fig.3 BERT Input Representation](figs/fig3_bertviz_input.jpeg)

