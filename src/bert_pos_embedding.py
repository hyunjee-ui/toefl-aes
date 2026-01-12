import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# data load
pos_df = pd.read_csv("/Users/hyunjee/project/toefl/data/processed/pos_sequence_only.csv")
pos_sequences = pos_df["pos_sequence"].fillna("")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Function to get BERT embeddings
def get_embedding(pos_seq):
    """POS 시퀀스 입력 -> [CLS] 토큰 임베딩 반환"""
    try:
        inputs=tokenizer(
            pos_seq,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs=model(**inputs)
            cls_embedding=outputs.last_hidden_state[:,0,:]
            return cls_embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error:{e}")
        return np.zeros(768)
    
# Generate embeddings
embeddings = []

save_partial_path=("/Users/hyunjee/project/toefl/data/embeddings/pos_embeddings_partial.npy")
final_path=("/Users/hyunjee/project/toefl/data/embeddings/pos_embeddings_final.npy")

for i, pos_seq in enumerate(
    tqdm(
        pos_sequences,
        desc="Generating BERT POS Embeddings"
    )
):
    emb = get_embedding(pos_seq)
    embeddings.append(emb)
    
    # Save partial embeddings every 1000 samples
    if (i + 1) % 1000 == 0:
        np.save(save_partial_path, np.stack(embeddings))
        print(f" Saved partial embeddings at index {i + 1} to {save_partial_path}")

# Save final embeddings
np.save(final_path, np.stack(embeddings))
print(f" All POS embeddings saved to {final_path}")