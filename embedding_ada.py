import openai
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
MODEL_NAME = config['openai']['Name']
openai.azure_endpoint = config['openai']['base']
openai.api_key = config['openai']['Key']
openai.api_type = 'azure'
openai.api_version = config['openai']['version']

def get_embedding(text, model_name=MODEL_NAME):
    response = openai.embeddings.create(
        input = text,
        model = MODEL_NAME
    )
    return response.data[0].embedding

if __name__ == '__main__':
    text = 'I love you 3000'
    print(get_embedding(text))