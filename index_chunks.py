# -*- codeing = utif-8 -*-
# @Time：2024/12/270:47
# @File:index_chunks.py
# @software:PyCharm
#在不同的chunk策略下构建不同的向量数据库
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from agentic_chunking import load_pdf,clean_pdf,extract_region
from process_data import remove_newlines
from args import schema_c,schema_d,schema_e
from process_data import txt_to_dictlist

# 1. 添加元数据，构建document对象
def chunk_metadata(dict_list, schema):
    '''
    :param chunks: [{},{},{}]
    :param schema: {"Page_content":[],"Metadata":[]}
    :return: new_chunks:Document对象
    '''
    new_chunks = []
    for chunk in dict_list:
        content_list = []
        meta_dict = {}
        for page in schema["Page_content"]:
            content_list.append(page + ':' + chunk[page])
        for meta in schema["Metadata"]:
            meta_dict[meta] = chunk[meta]
        new_chunk = Document(
            page_content="\n".join(content_list),
            metadata=meta_dict
        )
        new_chunks.append(new_chunk)
    return new_chunks
#2. 直接添加给定的元数据
def add_metadata(chunk_list,metadata_dict):
    new_chunks=[]
    for chunk in chunk_list:
        new_chunk = Document(
            page_content=chunk,
            metadata=metadata_dict
        )
        new_chunks.append(new_chunk)
    return new_chunks

# 3. 构建向量数据库索引
def index(db_path, new_chunks,embeddings):
    #embeddings = HuggingFaceBgeEmbeddings(model_name='TencentBAC/Conan-embedding-v1')
    if os.path.exists(db_path):
        vector_db = Chroma(persist_directory=db_path,
                           embedding_function=embeddings,
                           )
    else:
        vector_db = Chroma.from_documents(new_chunks, embeddings,
                                          persist_directory=db_path,
                                        )
    return vector_db

def pdf_page_index(pdf_path,db_path,embeddings):
    '''
    pdf_path:加载pdf的路径
    db_path:储存向量数据库的路径
    将pdf的每一页看做一个chunk，进行indexing
    '''
    documents = load_pdf(pdf_path)
    new_documents = clean_pdf(documents)
    new_docs = []
    for doc in new_documents:
        new_doc = remove_newlines(doc)
        new_docs.extend(new_doc)
    #print(new_docs)
    for doc in new_docs:
        metadata=doc.metadata
        if 'source' in metadata.keys() and 'region' not in metadata.keys():
            meta = extract_region(metadata)
            metadata['region'] = meta
    #print(new_docs)
    database = index(db_path,new_docs, embeddings)
    print(f"向量数据库已经成功保存至{db_path}")
    return database
#不同chunk策略
def overlap_chunks(document, chunk_size, overlap_size):
    '''
    Start by exploring a variety of chunk sizes, including smaller chunks (e.g., 128 or 256 tokens) for capturing more granular semantic information and larger chunks (e.g., 512 or 1024 tokens) for retaining more context.
    :param document:
    :param chunk_size: 128,256,512,1024
    :param overlap_size: 16,32,64,128
    :return:
    '''
    text = "".join([doc.page_content for doc in document])
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def chunksize_indexing(pdf_path,chunk_size,overlap,db_path,embeddings):
    #构建不同chunksize的向量数据库
    documents = load_pdf(pdf_path)
    new_documents = clean_pdf(documents)
    new_docs = []
    for doc in new_documents:
        metadata=doc[0].metadata
        metadata_dict={'region':extract_region(metadata)}
        new_doc = remove_newlines(doc)
        chunk=overlap_chunks(new_doc, chunk_size, overlap)#列表
        new_chunk=add_metadata(chunk, metadata_dict)
        new_docs.extend(new_chunk)

    database = index(db_path, new_docs, embeddings)
    print(f"向量数据库已经成功保存至{db_path}")
    return database

def agent_chunking(file_path,schema,db_path,embeddings):
    '''
    将精细抽取的知识块进行索引
    '''
    data = txt_to_dictlist(file_path)
    chunk_data = chunk_metadata(data, schema)
    #print(chunk_data)
    database = index(db_path, chunk_data, embeddings)
    print(f"向量数据库已经成功保存至{db_path}")
    return database


if __name__ == "__main__":
    '''
    基于TencentBAC 的embedding向量构建向量知识库为例
    '''

    load_dotenv()

    os.environ["LANGCHAIN_API_KEY"] = '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'
    os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 停止运用langsmith追踪
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "langchain_for_llm_application_development"

    pdf_path='data/'
    file_path=[
        'data/c_chunks_llm.txt',
        'data/d_chunks_llm.txt',
        'data/e_chunks_llm.txt',
    ]

    model_name = "TencentBAC/Conan-embedding-v1"
    model_kwargs = {'device': 'cpu',"trust_remote_code": True}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )

    #1. chunksize chunking
    print('*'*50+'chunksize chunking'+'*'*50)
    db_path = [
        'data/index_db/TencentBAC/chunk128_index',
        'data/index_db/TencentBAC/chunk256_index',
        'data/index_db/TencentBAC/chunk512_index',
        'data/index_db/TencentBAC/chunk1024_index'
    ]
   
    chunk_size=[128,256,512,1024]
    overlap=[16,32,64,128]
    for i in range(len(chunk_size)):
        chunksize_indexing(pdf_path,chunk_size[i],overlap[i],db_path[i],embeddings)

    #2. page chunking
    print('*'*50+'page chunking'+'*'*50)
    pdf_path='data/'
    db_path='data/index_db/TencentBAC/chunk_page_index'
    pdf_page_index(pdf_path,db_path,embeddings)

    #3. agent chunking
    print('*'*50+'agent chunking'+'*'*50)
    db_path = [
        'data/index_db/TencentBAC/chunk_c_index',
        'data/index_db/TencentBAC/chunk_d_index',
        'data/index_db/TencentBAC/chunk_e_index',
    ]
    schema=[
    schema_c,
    schema_d,
    schema_e 
    ]
    for i in range(len(db_path)):
        agent_chunking(file_path[i],schema[i],db_path[i],embeddings)
