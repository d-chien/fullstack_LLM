from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from hf_pipeline import get_sentiments_with_pipeline

model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_tokenizer.save_pretrained('nlp_models/distilbert-base')

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained('nlp_models/distilbert-base')

# use function from hf_pipeline.py
cls = get_sentiments_with_pipeline(
    model_name='nlp_models/distilbert-base',
    tokenizer_name='nlp_models/distilbert-base',
    #task = 'sentiment-analysis',
    #return_all_scores=True
    string_arr=["I love you 3000"]
)
print(cls)