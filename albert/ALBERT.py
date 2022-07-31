import torch
import torch.nn as nn
from Transformer import Transformer
from Embeddings import ALBERTTokenEmbedding
from typing import Optional

class ALBERT(nn.Module):
    def __init__(self, vocab_size:int, layer_iter:int = 12 , num_group:int = 1, token_embedding_size:int = 128,
                 model_hidden:int = 768, num_head:int =12, dropout:float=0.1):
        super().__init__()

        assert model_hidden % num_head == 0, "모델의 은닉 개수와 어텐션 헤드 수가 맞지않습니다."

        self.ALBERTTokenEmbedding = ALBERTTokenEmbedding(vocab_size)
        self.token_to_hidden_project_layer = nn.Linear(token_embedding_size, model_hidden) # 128 ,768
        self.layer_iter = layer_iter  # 12 회
        self.num_group = num_group # 뭔지 잘모름
        self.model_hidden = model_hidden
        self.num_head = num_head
        self.feed_forward_hidden= model_hidden * 4 # 피드포워드 인닉개수 H = E * 4라고 정의

        self.transformer_layer_group = nn.ModuleList(
            [Transformer(model_hidden, model_hidden *4, num_head , dropout) for _ in range(num_group)]
        )

    def forward(self,input_ids:torch.IntTensor, segment_ids:Optional[torch.IntTensor]=None,
                mask: Optional[torch.ByteTensor]=None)->torch.Tensor:

        if segment_ids is not None :
            assert input_ids.shape == segment_ids.shape, "Input_ids and Segment_ids shape MisMatching"
            token_embedding = self.ALBERTTokenEmbedding(input_ids,segment_ids)
        else :
            token_embedding = self.ALBERTTokenEmbedding(input_ids)
        # Transformer Encoder 에 Input
        input_hidden = self.token_to_hidden_project_layer(token_embedding)

        for i in range(self.layer_iter):
            """
                num_group 이 있는 이유
                 
                num_group = 1 일때는 모든 12개의 트랜스 포머가 파라미터를 공유( transformer_layer_group[0])의 인코더만 사용하니까 
                num_group = 2 일때는 [0,0,0,0,0,1,1,1,1,1,1] -> group[0],group[1] 두개의 층만 사용 
                            num_group이 커질수록 cross-layer sharing Prameter를 줄이게 됌 
                            
            
            """

            group_index = (i * self.num_group) // self.layer_iter
            # group_index = i % self.num_group
            if i == 0 :
                x = self.transformer_layer_group[group_index](input_hidden,mask)
            else :
                x = self.transformer_layer_group[group_index](x,mask)

        return x