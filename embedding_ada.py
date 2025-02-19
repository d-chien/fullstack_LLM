import openai
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
MODEL_NAME = config['openai']['Name']
openai.azure_endpoint = config['openai']['base']
openai.api_key = config['openai']['Key']
openai.api_type = 'azure'
openai.api_version = config['openai']['version']

response = openai.embeddings.create(
    input = "I love you 3000",
    model = MODEL_NAME
)
print(response.data[0].embedding)