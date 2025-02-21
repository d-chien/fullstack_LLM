import numpy as np
import faiss
from embedding_ada import get_embedding

def add_to_faiss_index(embeddings):
    vector = np.array(embeddings)
    index = faiss.IndexFlatL2(vector.shape[1])   # faiss not support cosine similarity, so we use L2
    index.add(vector)
    return index

# embedding the question, then search in FAISS, return a list of tuple, (text, distance)
def vector_search(index, query_embedding, text_array, k=1):
    dist, indices = index.search(np.array([query_embedding]), k)
    print(f'dist: {dist}')
    print(f'indices: {indices}')
    return [(text_array[i], float(dist)) for dist, i in zip(dist[0],indices[0])]

def main():
    text_array = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    embeddings = [get_embedding(text) for text in text_array]
    faiss_index = add_to_faiss_index(embeddings)

    query = 'I love you'
    query_embedding = get_embedding(query)
    result = vector_search(faiss_index, query_embedding, text_array,k=2)
    print(f'search {query}: {result}')

if __name__ == '__main__':
    main()