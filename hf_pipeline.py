from transformers import pipeline

def get_sentiments_with_pipeline(model_name, tokenizer_name, string_arr):
    # initialize the pipeline
    ppl = pipeline(
        task='sentiment-analysis',
        model=model_name,
        tokenizer=tokenizer_name,
        return_all_scores=True
    )

    # get result
    results = ppl(string_arr)
    return results

if __name__ == '__main__':
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    string_arr = ["我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離",
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    results = get_sentiments_with_pipeline(model_name, tokenizer_name, string_arr)
    print(results)