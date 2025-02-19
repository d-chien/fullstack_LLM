from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_sentiments(model_name, string_arr):
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('tokenizer:',tokenizer,'\n')

    # Tokenize input
    inputs = tokenizer(string_arr, padding=True, truncation=True, return_tensors="pt")
    print('inputs:',inputs,'\n')

    # initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print('model:',model,'\n')
    assert model.config.output_attentions == False

    # make predictions
    outputs = model(**inputs)
    print('outputs:',outputs,'\n')

    # convert logits to probabilities with softmax
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print('predictions:',predictions,'\n')

    return predictions

if __name__ == '__main__':
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    string_arr = ["我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離",
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    predictions = get_sentiments(model_name, string_arr)
    print(predictions)