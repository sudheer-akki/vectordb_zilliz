import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def convert_embeddings(text, 
                       tokenizer, 
                       embed_model, 
                       device = "cuda"):
    encoded_input = tokenizer(text, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    encoded_input = encoded_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = embed_model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    vector_embeddings = sentence_embeddings.cpu().numpy().tolist()
    return vector_embeddings