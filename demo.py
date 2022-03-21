# from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
# #from haystack.document_store import ElasticsearchDocumentStore

# from haystack.retriever.dense import EmbeddingRetriever
# from haystack.utils import print_answers, launch_es
# import pandas as pd
# import requests
# import logging
# import subprocess
# import time

# launch_es()

# document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
#                                             index="document",
#                                             embedding_field="question_emb",
#                                             embedding_dim=384,
#                                             excluded_meta_data=["question_emb"],
#                                             similarity="cosine")

# # Download a csv containing some FAQ data
# # Here: Some question-answer pairs related to RCE
# temp = requests.get("https://raw.githubusercontent.com/paras-j/EP_FAQs/main/RCE_FAQs.csv")
# open('faq_rce.csv', 'wb').write(temp.content)

# # Get dataframe with columns "question", "answer" and some custom metadata
# df = pd.read_csv("faq_rce.csv")
# # Minimal cleaning
# df.fillna(value="", inplace=True)
# df["question"] = df["question"].apply(lambda x: x.strip())
# print(df.head())

# # Get embeddings for our questions from the FAQs
# questions = list(df["question"].values)
# df["question_emb"] = retriever.embed_queries(texts=questions)
# df = df.rename(columns={"question": "text"})

# # Convert Dataframe to list of dicts and index them in our DocumentStore
# docs_to_index = df.to_dict(orient="records")
# document_store.write_documents(docs_to_index)


# @st.cache(allow_output_mutation=True)
# def get_retriever():
#     retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True)
#     return retreiver


# #Initialize a Pipeline (this time without a reader) and ask questions
# from haystack.pipeline import FAQPipeline
# pipe = FAQPipeline(retriever=get_retriever())

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")
    
# text = st.text_input('Enter your questions here....')
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#       prediction = pipe.run(query="How is the virus spreading?", top_k_retriever=1)
#       st.write('answer: {}'.format(print_answers(prediction, details="all")))
#     st.write("")

    
    
    
    
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
import pandas as pd
import requests

import os
from subprocess import Popen, PIPE, STDOUT

#urllib.request.urlopen('https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz')

import wget
url = 'https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz'
print("I am at 81")
import tarfile
filename = tarfile.open(wget.download(url))
filename.extractall('./')
filename.close()
print("I am at 86")
#! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
#! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
#! chown -R daemon:daemon elasticsearch-7.9.2
# import os
# from subprocess import Popen, PIPE, STDOUT
es_server = Popen(["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon)
print("I am at 93")
import time
time.sleep(30)
print("I am at 96")
from haystack.document_stores import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index="document",
    embedding_field="question_emb",
    embedding_dim=384,
    excluded_meta_data=["question_emb"],
)
print("I am at 107")
@st.cache(allow_output_mutation=True)
def get_document_store():
  temp_rce = requests.get("https://raw.githubusercontent.com/paras-j/EP_FAQs/main/RCE_FAQs.csv")
  open("faq_rce.csv", "wb").write(temp_rce.content)
  import chardet
  with open('faq_rce.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
  df = pd.read_csv('faq_rce.csv', encoding=result['encoding'])
  # Minimal cleaning
  df.fillna(value="", inplace=True)
  df["question"] = df["question"].apply(lambda x: x.strip())
  print(df.head())
  # Get embeddings for our questions from the FAQs
  questions = list(df["question"].values)
  df["question_emb"] = retriever.embed_queries(texts=questions)
  df = df.rename(columns={"question": "content"})
  # Convert Dataframe to list of dicts and index them in our DocumentStore
  docs_to_index = df.to_dict(orient="records")
  document_store.write_documents(docs_to_index)
  return document_store
print("I am at 128")
@st.cache(allow_output_mutation=True)
def get_retriever():
  retriever = EmbeddingRetriever(document_store=get_document_store(), embedding_model="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True)
  return retriever

from haystack.pipelines import FAQPipeline
pipe = FAQPipeline(retriever=get_retriever())

from haystack.utils import print_answers

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Question Answering Webapp")
st.text("What would you like to know today?")
    
text = st.text_input('Enter your questions here....')
if text:
    st.write("Response:")
    with st.spinner('Searching for answers....'):
      prediction = pipe.run(query=text, params={"Retriever": {"top_k": 1}})
      st.write('answer: {}'.format(print_answers(prediction, details="medium")))
    st.write("")                  
