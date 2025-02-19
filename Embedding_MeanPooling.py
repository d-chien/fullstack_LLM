from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# word to embedding
sentences = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
             'baby shark doo doo doo doo doo doo',
             'I love you 3000',
             '耶和華祝福滿滿，就像海邊的沙。',
             '臨表涕泣，不知所云。']

# load model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# tokenize
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# there will be three part of the output: input_ids, attention_mask, token_type_ids, and later we will use attention_mask for futher calculation

# calculate the embeddings
with torch.no_grad():   # torch.no_grad() prevents tracking history (and using memory) for the forward pass, no gradient applied
    model_output = model(**encoded_input)

# defined the mean pooling function
def mean_pooling(model_output, attention_mask):
    # first, take out embedding text, model_output[0] is the last_hidden_state, which is the output of the last layer of the model
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings

    # then, take out the attention_mask, and add a dimension to it, so that it can be multiplied with the token_embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # count actual number of tokens of each sentence, use torch.clamp to make sure the minimum value is 1e-9
    word_length = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    out = torch.sum(token_embeddings * input_mask_expanded, 1) / word_length
    return out

# get the mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# L2 normalization, with dim=1, which means normalize each row
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print(f'sentence_embeddings: \n{sentence_embeddings}')