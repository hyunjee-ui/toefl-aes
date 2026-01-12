import pandas as pd #csv 파일 읽어들이고 데이터 처리하는 라이브러리 
import numpy as np #임베딩을 배열 형태로 저장 
import torch #PyTorch 사용하여 모델 로딩, GPU에서 실행 
from transformers import BertTokenizer, BertModel #허깅페이스에서 BERT 불러오기
from tqdm import tqdm #진행 상태를 표시하는 그로그레스 바 제공하는 라이브러리 
import time #시간이 필요한 작업을 기록할 때 사용 


df = pd.read_csv("/Users/hyunjee/project/toefl/data/processed/output.csv") #임베딩을 추출할 텍스트 데이터를 포함하는 데이터프레임 

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #BERT 모델의 토크나이저 불러오기 -> 텍스트 데이터를 토큰화 
model=BertModel.from_pretrained("bert-base-uncased") #-> 텍스트에 대해 임베딩 생성 가능 

device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_embedding(text): #텍스트를 입력받아 BERT 모델을 사용해 임베딩 추출
    try:
        inputs=tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs=model(**inputs) #model(**inputs)로 토크나이즈된 텍스트를 입력

            return outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy() #[CLS] 토큰에 해당하는 임베딩 반환
    except Exception as e:
        print(f"⚠️ Error processing text: {e}")
        return np.zeros(768) #추출한 벡터를 Numpy 배열로 변환하여 리턴 
    
embeddings=[]
save_path="/Users/hyunjee/project/toefl/data/embeddings/embeddings_partial.npy" #임시 저장 경로

for i, text in enumerate(tqdm(df["content"],desc="BERT 임베딩 추출 중")):
    emb = get_embedding(text)
    embeddings.append(emb)

    if(i+1)%100 ==0:
        np.save(save_path, np.stack(embeddings))
        print(f"{i+1}개까지 임시 저장 완료")
final_path = "/Users/hyunjee/project/toefl/data/embeddings/embeddings_final.npy"
np.save(final_path, np.stack(embeddings))
print(f" 모든 임베딩 완료: {final_path}")