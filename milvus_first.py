from pymilvus import connections, db, DataType, FieldSchema, CollectionSchema, utility, Collection
from embedding_ada import get_embedding

# establish database
connection = connections.connect(
    'default',
    host='localhost',
    port='19530',
)
if 'lyrics' not in  db.list_database():
    database = db.create_database('lyrics')
print(db.list_database())

# setup embedding
def prepare_embeddings(text_array):
    embeddings = [get_embedding(text) for text in text_array]
    return embeddings

# establish schema and collection
# index_params =  HNSW, COSINE
# Maximum degree of the node
def create_collection(collection_name):
    fields = [
        FieldSchema(name = 'id', dtype = DataType.INT64, is_primary = True, description='Ids', auto_id = False),
        FieldSchema(name = 'lyric', dtype = DataType.VARCHAR, max_length = 500, description='lyric texts'),
        FieldSchema(name = 'embedding', dtype = DataType.FLOAT_VECTOR, dim = 1536, description='Embedding vectors')
    ]
    schema = CollectionSchema(fields = fields, description = 'lyrics collection')
    collection = Collection(name = collection_name, schema = schema)

    index_params = {
        'index_type': 'HNSW',
        'metric_type': 'COSINE',
        'params': {'M': 16, 'efConstruction': 500}
    }
    collection.create_index(field_name = 'embedding', index_params = index_params)
    return collection

# insert vector data
def insert_to_milvus(collection, text_array, embedding_array):
    entities = [
        {'id': i, 'lyric': text_array[i], 'embedding': embedding_array[i]}
        for i in range(len(text_array))
    ]
    collection.insert(data = entities)

# search function
def search_from_milvus(collection, query_embedding, k=1):
    search_params = {
        'metric_type':'COSINE'
    }

    results = collection.search(
        data = [query_embedding],
        anns_field = 'embedding', #search in embedding field
        limit = k,
        param = search_params,
        output_fields = ['lyric']
    )
    print(f'results[0]: \n{results[0]}')
    ret = []
    for hit in results[0]:
        row = []
        row.extend([hit.id, hit.score, hit.entity.get('lyric')])
        ret.append(row)

    return ret

def main(connection, collection):
    COLLECTION_NAME = 'lyrics_collection'

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    collection = create_collection(COLLECTION_NAME)
    collection.load()

    text_array = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離', 
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    embeddings = prepare_embeddings(text_array)

    insert_to_milvus(collection, text_array, embeddings)

    query = 'I love you'
    query_embedding = get_embedding(query)
    result = search_from_milvus(collection, query_embedding)
    print(f'search {query}: {result}')

if __name__ == '__main__':
    connection = connections.connect(
        'default',
        host='localhost',
        port='19530',
    )
    if utility.has_collection('lyrics_collection'):
        collection = Collection('lyrics_collection')
        collection.load()
    else:
        collection = None
    main(connection, collection)