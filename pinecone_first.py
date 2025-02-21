from pinecone import Pinecone
from configparser import ConfigParser
import time
from embedding_ada import get_embedding
config = ConfigParser()
config.read('config.ini')
api_key = config['pinecone']['Key']
index_name = config['pinecone']['index_name']

# initialize pinecone
def init_pinecone(index_name):
    pc = Pinecone(api_key=api_key, environment='starter')
    index = pc.Index(name=index_name)
    return index



# use to create index in pinecone
def create_index(index_name):
    if index_name not in Pinecone.list_indexes():
        Pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
        while not Pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f'Index {index_name} created')

# add embeddings to pinecone
def add_to_pinecone(index, embeddings, text_array):
    ids = [str(i) for i in range(len(embeddings))]
    embeddings = [embedding for embedding in embeddings]
    text_to_meta = [{'content':text} for text in text_array]
    tuple_id_embedding_meta = zip(ids, embeddings, text_to_meta)
    index.upsert(vectors=tuple_id_embedding_meta)
    print('Embeddings added to Pinecone')

# search embeddings from pinecone
def search_from_pinecone(index, query_embedding, k=1):
    results = index.query(vector=[query_embedding], top_k=k, include_metadata=True)
    return results


def main():
    text_array = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    embeddings = [get_embedding(text) for text in text_array]

    index = init_pinecone(index_name)

    # upload data to pinecone
    # add_to_pinecone(index, embeddings, text_array)

    # search data from pinecone
    query = 'I love you'
    query_embedding = get_embedding(query)
    result = search_from_pinecone(index, query_embedding)

    print(f'search {query}: {result}')


    


if __name__ == '__main__':
    main()