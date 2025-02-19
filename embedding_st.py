from sentence_transformers import SentenceTransformer
sentence = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
            'baby shark doo doo doo doo doo doo',
            'I love you 3000',
            '耶和華祝福滿滿，就像海邊的沙。',
            '臨表涕泣，不知所云。']
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentence)
print(f'sentence_embeddings: \n{sentence_embeddings}')