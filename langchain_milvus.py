import openai
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def get_embedding(config):
    embeddings = AzureOpenAIEmbeddings(
        api_key=config['openai']['Key'],
        model = config['openai']['Name'],
        # openai_api_type = openai.api_type,
        openai_api_type=config['openai']['version'],
        azure_endpoint=config['openai']['base'],
        chunk_size = 1
    )
    return embeddings

def get_doc_store(texts, embeddings):
    doc_store = Milvus.from_texts(
        texts = texts,
        embedding = embeddings,
        connection_args = {
            'uri': 'http://localhost:19530',
        },
        drop_old = True,
        collection_name = 'PDF_langchain'
    )
    return doc_store

def load_and_split_doc(filepath='./pdf/qa.pdf'):
    loader = PyPDFLoader(filepath)
    text = loader.load()
    splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    docs = splitter.split_documents(text)
    texts = [doc.page_content for doc in docs]
    return texts

def get_chat_model(config):
    chat_model = AzureChatOpenAI(
        azure_deployment= config['gpt']['Name'],
        api_key = config['gpt']['Key'],
        openai_api_type = 'azure',
        azure_endpoint = config['gpt']['base'],
        api_version = config['gpt']['version']
    )
    return chat_model

def ask_question_with_context(qa,question, chat_history):
    # query = ""
    result = qa.invoke({'input': question, 'chat_history': chat_history})
    print(f'answer: {result.answer}')
    chat_history = []
    chat_history.append((question, result.answer))
    return chat_history

def conversational_retrieval(doc_store, chat_model):
    retriever = doc_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name='context')
    ])
    document_chain = create_stuff_documents_chain(chat_model, prompt)

    def retriever_with_message_conversion(inputs):
      query = inputs["input"]
      chat_history = inputs["chat_history"]
      retrieved_docs = retriever.get_relevant_documents(query)
      context_messages = [AIMessage(content=doc.page_content) for doc in retrieved_docs]
      return {"context": context_messages} #修正此處

    retriever_runnable = RunnableLambda(retriever_with_message_conversion) #將function 轉換為 runnable
    retriever_chain = create_history_aware_retriever(chat_model, retriever_runnable, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain

def main(config):
    embeddings = get_embedding(config)
    docs = load_and_split_doc()
    doc_store = get_doc_store(docs, embeddings)
    chat_model = get_chat_model(config) #config需要傳入
    chat_history = []
    qa = conversational_retrieval(doc_store, chat_model)

    while True:
        question = input("Ask a question: ")
        if question == "exit":
            break
        chat_history = ask_question_with_context(qa, question, chat_history)


if __name__=='__main__':
    main(config)