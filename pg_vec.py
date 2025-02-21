from embedding_ada import get_embedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from models import Embeddings

# create a new connection of PostgreSQL
engine = create_engine('postgresql://user:password@localhost:5432/vector')
Session = sessionmaker(bind=engine)
session = Session()

# add to vec
def add_to_pg_vec(session, embeddings):
    embeddings_obj = [Embeddings(vector = vector) for vector in embeddings]
    session.add_all(embeddings_obj)
    session.commit()

# cosine similarity search
def search_from_pg_vec(session, text_array, query_embedding, k=1):
    results = session.scalars(select(Embeddings).order_by(Embeddings.vector.cosine_distance(query_embedding)).limit(k))
    distance = session.scalars(select(Embeddings.vector.cosine_distance(query_embedding)))

    # zip dist and text_array and trasform cosine distance to similarity
    # results.id is 1-based index, so we need to minus 1 to get the 0-based index
    return [(text_array[result.id-1], 1 - float(dist)) for result, dist in zip(results, distance)]

def main():
    text_array = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
                  'baby shark doo doo doo doo doo doo',
                  'I love you 3000',
                  '耶和華祝福滿滿，就像海邊的沙。',
                  '臨表涕泣，不知所云。']
    embeddings = [get_embedding(text) for text in text_array]
    pg_vector = add_to_pg_vec(session, embeddings)

    query = 'I love you'
    query_embedding = get_embedding(query)
    result = search_from_pg_vec(session, text_array, query_embedding)
    print(f'search {query}: {result}')

if __name__ == '__main__':
    try:
        main()
    finally:
        session.close()
        engine.dispose()