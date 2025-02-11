# -*- codeing = utif-8 -*-
# @Time：2024/12/2916:12
# @File:e_retriever.py
# @software:PyCharm
import os
import re
import jieba
from langchain.retrievers.self_query.base import SelfQueryRetriever
from args import metadata_field_info,document_content_description
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agentic_chunking import extract_region
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from bert_score import BERTScorer
from process_data import txt_to_dictlist,text_to_document
from llm_chain import llm_agent_s
from promt_chain import prompt_requery

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 停止运用langsmith追踪
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain_for_llm_application_development"


def dense_retriver(query,vector_db,topk):
    '''
    运用密集检索器
    :param query:
    :param vector_db:
    :param topk:
    :return:
    '''
    retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": topk})
    topk_docs=retriever.invoke(query)
    return topk_docs

def bm25_retriver(query,vector_db,num,topk):
    '''
    运用稀疏检索器BM25
    :param query:
    :param vector_db:向量数据库
    :param topk:
    :return:
    '''
    def cut_words(text):

        return jieba.lcut(text)
    
    document=dense_retriver('',vector_db,num)
    sparse_retriever=BM25Retriever.from_documents(document,preprocess_func=cut_words,k=topk)
    topk_docs = sparse_retriever.invoke(query)
    return topk_docs


def metadata_index(docs):
    '''

    :param docs: document对象
    :return:
    '''
    metadata_with_index = []
    for index, doc in enumerate(docs):
        metadata=doc.metadata
        k_list =list(metadata.keys())
        if len(k_list)==1 and 'region' in k_list:
            combined_content=doc.metadata['region']
        elif len(k_list)==3 and 'region' in k_list:
            combined_content = doc.metadata['region']
        else:
            combined_content = f"{doc.metadata['地区']}{doc.metadata['预警信号']}"
        new_metadata = {'index': index}
        metadata_with_index.append(
            Document(
                metadata=new_metadata,
                page_content=combined_content
            )
        )
    return metadata_with_index

def hybrid_retriver1(query,vector_db,k1,k2,api):
    #用于评估时用，参数多变
    topk_docs = dense_retriver(query, vector_db,k1)
    metadata_with_index = metadata_index(topk_docs)
    sparse_retriever=BM25Retriever.from_documents(metadata_with_index,k=k2)
    sub_query=llm_agent_s(prompt_requery,{'query':query},api)
    print(sub_query)
    candidate_info = sparse_retriever.invoke(sub_query)
    c_info_index = candidate_info[0].metadata['index']
    gold_doc = topk_docs[c_info_index]
    return gold_doc


def hybrid_retriver2(query,vector_db,topk):
    #用于实际生成时用，参数固定
    topk_docs = dense_retriver(query, vector_db,topk)
    metadata_with_index = metadata_index(topk_docs)
    sparse_retriever=BM25Retriever.from_documents(metadata_with_index,k=1)
    candidate_info = sparse_retriever.invoke(query)
    c_info_index = candidate_info[0].metadata['index']
    gold_doc = topk_docs[c_info_index].page_content
    return gold_doc



 