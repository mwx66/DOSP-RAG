# -*- codeing = utif-8 -*-
# @Time：2024/12/2718:33
# @File:e_chunks.py
# @software:PyCharm
#不同的chunk构造对retrieve的影响
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from process_data import txt_to_dictlist
from bert_score import BERTScorer
from e_retriever import dense_retriver,bm25_retriver,hybrid_retriver1

def bert_re_score(gold_info, re_doc,source='none'):
    '''
    针对chunk数据，每个chunk都是document对象，具有元数据
    :param gold_info: 正确的文档(从qa_data的answer)
    :param re_doc: 检索到的文档，document对象，包括page_content与metadata
    :param source: 文档来源
    :return: 相关性分数：将bert_socre中F1作为相关性分数
    '''
    #数据准备
    re_context = [re_doc.page_content]
    metadata = re_doc.metadata
    gold_context=[gold_info['answer']]
    #元数据过滤
    if source != 'none' and source not in metadata.values():
        return 0
    #模型加载,相关性分数计算
    scorer = BERTScorer(lang='zh')
    P, R, F1 = scorer.score(re_context, gold_context)
    return F1

def precision(k,r_score):
    '''
    Precision@k = 前k个结果中相关文档的数量 / k
    r_score：所有文档的相关性得分总和
    '''
    return r_score/k

def recall(num,r_score):#num的决定很重要，与chunk有关
    '''
    Recall@k = 前k个结果中相关文档的数量 / 知识库中相关文档总数
    '''
    return min(1,r_score/num)

def F1(p,r):
    '''
     F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    :return:
    '''
    return 2*(p*r)/(p+r)

def chunk_re_e(db_path, qa_path, s_path, k,num,embedding):

    sou_list = txt_to_dictlist(s_path)
    qa_data = txt_to_dictlist(qa_path)
    db = Chroma(persist_directory=db_path, embedding_function=embedding)

    p_score = []
    r_score = []
    for i in range(len(qa_data)):
        data = qa_data[i]
        query = data['query']
        docs = dense_retriver(query, db, k)
        re_score = 0
        for doc in docs:
            re_score += bert_re_score(data, doc, sou_list[i].get('source'))
        #p_score.append(precision(k, re_score))
        r_score.append(recall(num, re_score))

    #print(f"precision@{k}", sum(p_score) / len(p_score))
    print(f"recall@{k}:", sum(r_score) / len(r_score))

def chunk_re_e1(db_path, qa_path, s_path, k,num,embedding):

    sou_list = txt_to_dictlist(s_path)
    qa_data = txt_to_dictlist(qa_path)
    db = Chroma(persist_directory=db_path, embedding_function=embedding)

    p_score = []
    r_score = []
    for i in range(len(qa_data)):
        data = qa_data[i]
        query = data['query']
        #docs1 = dense_retriver(query, db, k)
        docs=bm25_retriver(query,db,241,k)
        re_score = 0
        for doc in docs:
            re_score += bert_re_score(data, doc, sou_list[i].get('source'))
        p_score.append(precision(k, re_score))
        r_score.append(recall(num, re_score))

    print(f"precision@{k}", sum(p_score) / len(p_score))
    print(f"recall@{k}:", sum(r_score) / len(r_score))

def chunk_re_e2(db_path, qa_path, s_path, k,embedding):

    sou_list = txt_to_dictlist(s_path)
    qa_data = txt_to_dictlist(qa_path)
    db = Chroma(persist_directory=db_path, embedding_function=embedding)

    for i in range(len(qa_data)):
        
        data = qa_data[i]
        query = data['query']
        print('*'*120)
        print('query:',query)
        docs1 = dense_retriver(query, db, k)
        print(' docs1:',docs1[0].metadata)
        p_score1=bert_re_score(data, docs1[0], sou_list[i].get('source'))

        docs=hybrid_retriver1(query,db,k,k,'ZZZ_MODEL_1')
        print(' docs:',docs.metadata)
        p_score2=bert_re_score(data, docs, sou_list[i].get('source'))

    print(f"dense_precision@{k}{1}", p_score1)
    print(f"hybrid_precision@{k}{1}", p_score2)



if __name__ == "__main__":
    #在agent chunking 的设置下，评估不同检索策略下的检索表现

    # k=3,5,7,10
    load_dotenv()
    os.environ["LANGCHAIN_API_KEY"] = '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'
    os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 停止运用langsmith追踪
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "langchain_for_llm_application_development"

    embeddings=[
        'moka-ai/m3e-large',
        'BAAI/bge-large-zh-v1.5',
        'thenlper/gte-large-zh',
        'jinaai/jina-embeddings-v2-base-zh',
        'TencentBAC/Conan-embedding-v1'
    ]

    name=[
        'Mokaai',
        'BAAI',
        'GTE',
        'Jinaai',
        'Tencent'
          ]

    qa_list=[
        'data/c_qa.txt',
        'data/d_qa.txt',
        'data/e_qa.txt'
    ]

    s_list=[
        'data/c_source.txt',
        'data/d_source.txt',
        'data/e_source.txt'
        ]
    att_list=['分类','防御指南','标准']

    j=3#0,1,2,3,4
    r=2#0,1,2
    #i=0#0,1,2

    db_list=[

        'data/index_db/'+f"{name[j]}"+'/chunk_c_index',#79
        'data/index_db/'+f"{name[j]}"+'/chunk_d_index',#241
        'data/index_db/'+f"{name[j]}"+'/chunk_e_index',#241

    ]

    model_name = embeddings[j]
    model_kwargs = {'device': 'cpu',"trust_remote_code": True}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    for k in [5,10]:
        chunk_re_e2(db_list[r], qa_list[r], s_list[r], k, embedding)

'''
#1. 不同分块策略下，评估不同检索器的评估表现
embeddings=[
        'moka-ai/m3e-large',
        'BAAI/bge-large-zh-v1.5',
        'thenlper/gte-large-zh',
        'jinaai/jina-embeddings-v2-base-zh',
        'TencentBAC/Conan-embedding-v1'
    ]

    name=[
        'Mokaai',
        'BAAI',
        'GTE',
        'Jinaai',
        'Tencent'
          ]

    num_list=[2,4,3,2,1,1,1,1]

    qa_list=[
        'data/c_qa.txt',
        'data/d_qa.txt',
        'data/e_qa.txt'
    ]

    s_list=[
        'data/c_source.txt',
        'data/d_source.txt',
        'data/e_source.txt'
        ]
    att_list=['分类','防御指南','标准']

    j=4#0,1,2,3,4
    r=1#0,1,...,7
    i=0#0,1,2

    db_list=[

        'data/index_db/'+f"{name[j]}"+'/chunk_page_index',#186
        'data/index_db/'+f"{name[j]}"+'/chunk128_index',#623
        'data/index_db/'+f"{name[j]}"+'/chunk256_index',#313
        'data/index_db/'f"{name[j]}"+'/chunk512_index',#157
        'data/index_db/'+f"{name[j]}"+'/chunk1024_index',#79
        'data/index_db/'+f"{name[j]}"+'/chunk_c_index',#79
        'data/index_db/'+f"{name[j]}"+'/chunk_d_index',#241
        'data/index_db/'+f"{name[j]}"+'/chunk_e_index',#241

    ]

    model_name = embeddings[j]
    model_kwargs = {'device': 'cpu',"trust_remote_code": True}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    
    #1,3,5,10
    print('#'*50+f"{embeddings[j]}"+'#'*50)
    for i in [0,1,2]:
        r=i+5
        print('*' * 50 + f"数据库：{db_list[r]}-属性：{att_list[i]}" + '*' * 50)
        for k in [1,3,5,10]:
            chunk_re_e(db_list[r], qa_list[i], s_list[i], k, num_list[r],embedding)

'''
    
 
 
    
            
                















