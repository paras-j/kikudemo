from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
import pandas as pd
import requests

import os
from subprocess import Popen, PIPE, STDOUT

import wget
url = 'https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz'
wget.download(url)
#! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
#! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
#! chown -R daemon:daemon elasticsearch-7.9.2
# import os
# from subprocess import Popen, PIPE, STDOUT
es_server = Popen(
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
)

import time
time.sleep(30)

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
