import configparser
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Milvus

config = configparser.ConfigParser()
config.read('config.ini')
MODEL_NAME = config['openai']['Name']
openai.azure_endpoint = config['openai']['base']
# openai.api_key = config['openai']['Key']
openai.api_type = 'azure'
openai.api_version = config['openai']['version']

embeddings = AzureOpenAIEmbeddings(
    api_key=config['openai']['Key'],
    model = MODEL_NAME,
    # openai_api_type = openai.api_type,
    openai_api_type=openai.api_version,
    azure_endpoint=openai.azure_endpoint,
    chunk_size = 1
)

text_array = ['我會披星戴月的想你，我會奮不顧身的前進，遠方燈火愈來愈唏噓，凝視前方身後的距離',
              'baby shark doo doo doo doo doo doo',
              'I love you 3000',
              '耶和華祝福滿滿，就像海邊的沙。',
              '臨表涕泣，不知所云。']

doc_store = Milvus.from_texts(
    texts = text_array,
    embedding = embeddings,
    connection_args = {
        'uri': 'http://localhost:19530',
    },
    drop_old = True,
    collection_name = 'lyrics_collection'
)

question = 'I love you'
docs = doc_store.similarity_search_with_score(question)

document, score = docs[0]
print(document.page_content)
print(f'\nScore: {score}')