# -*- codeing = utif-8 -*-
# @Time：2024/12/271:10
# @File:e_genews.py
# @software:PyCharm
#对生成的新闻进行评估
from llm_chain import llm_agent_s
from promt_chain import prompt_generation,promt_gen_q1,promt_gen_q2,prompt_query1
from e_retriever import o_dual_retriver
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from process_data import txt_to_dictlist,textlist_to_json

def llm_new_gen(signal,policy,api_):
    input_={'signal':signal,'policy':policy}
    response=llm_agent_s(promt_gen_q1,input_,api_)
    return response

def llm_new_gen2(query,api_):
    input_={'query':query}
    response=llm_agent_s(promt_gen_q2,input_,api_)
    return response

def dkg_llm_gen(query,vb,dynamic_info,api_1,api_2):
    sub_query=llm_agent_s(prompt_query1,{'query':query},api_1)
    print('sub_query:',sub_query)
    defense =o_dual_retriver(sub_query,vb,5)
    print('defense:',defense)
    #signal = dynamic_info['dynamic_info']
    with open('answer/example2.txt', "r", encoding="utf-8") as f:
        example = f.read()
    input_={'signal':dynamic_info,'defense':defense,'example':example}
    response = llm_agent_s(prompt_generation, input_, api_2)
    return response

def gen_news1(out_file,model,flag):
        q_list=txt_to_dictlist('data/gen_e/query.txt')
        d_list=txt_to_dictlist('data/gen_e/dk.txt')
        loader=PyPDFLoader('data/中国气象局.pdf')
        document=loader.load()
        policy=''
        for doc in document:
            policy+='\n'+doc.page_content
        print('*'*50+f"政策文件"+'*'*50)
        #print('policy:',policy)
        print('policy:',len(policy))
        gen=[]
        if flag:
            for i in range(len(q_list)):
                query=q_list[i].get(str(i+1))
                signal=d_list[i].get(str(i+1))
                response=llm_new_gen(signal,policy,model)
                print('*'*50+f"第{i+1}条新闻"+'*'*50)
                print('query:',query)
                print('response:',response)
                gen.append(response)
            textlist_to_json(out_file,gen)
        else:
            embeddings = HuggingFaceBgeEmbeddings(model_name='TencentBAC/Conan-embedding-v1')
            db_path='data/index_db/TencentBAC/chunk_d_index'
            vb = Chroma(persist_directory=db_path,
                       embedding_function=embeddings,
                       )
            gpt4omini='ZZZ_MODEL'
            for i in range(len(q_list)):
                query=q_list[i].get(str(i+1))
                dynamic_info=d_list[i].get(str(i+1))
                print('*'*50+f"第{i+1}条新闻"+'*'*50)
                print('query:',query)
                response=dkg_llm_gen(query,vb,dynamic_info,gpt4omini,model)
                print('response:',response)
                gen.append(response)
            textlist_to_json(out_file,gen)

def gen_news2(out_file,model):
        q_list=txt_to_dictlist('data/gen_e/query.txt')
        d_list=txt_to_dictlist('data/gen_e/dk.txt')
        gen=[]
        for i in range(20):#len(q_list)
                query=q_list[i].get(str(i+1))
                response=llm_new_gen2(query,model)
                print('*'*50+f"第{i+1}条新闻"+'*'*50)
                print('query:',query)
                print('response:',response)
                gen.append(response)
        textlist_to_json(out_file,gen)

if __name__ == "__main__":

    m_list=['GPT4O','GPT4','CLAUDE3.5','GEMINI1.5','CHATGLM','QWEN2.5','DOUBAO']
    out_file1=[
        'data/gen_e/gen_examples/gpt4o_zh1.json',
        'data/gen_e/gen_examples/gpt4_zh1.json',
        'data/gen_e/gen_examples/claude3.5_zh1.json',
        'data/gen_e/gen_examples/gemini1.5_zh1.json',
        'data/gen_e/gen_examples/chatglm_zh1.json',
        'data/gen_e/gen_examples/qwen2.5_zh1.json',
        'data/gen_e/gen_examples/doubaow_zh1.json',
        'data/gen_e/gen_examples/gpt4omini_DOSPRAG_zh.json'
    ]

    out_file2=[
        'data/gen_e/gen_examples/gpt4o_zh2.json',
        'data/gen_e/gen_examples/gpt4_zh2.json',
        'data/gen_e/gen_examples/claude3.5_zh2.json',
        'data/gen_e/gen_examples/gemini1.5_zh2.json',
        'data/gen_e/gen_examples/chatglm_zh2.json',
        'data/gen_e/gen_examples/qwen2.5_zh2.json',
        'data/gen_e/gen_examples/doubaow_zh2.json',
    ]

    x=6
    print('*'*50+f"模型{m_list[x]}的新闻生成"+'*'*50)
    #1. 给定问询、动态信息与政策文件的新闻生成
    gen_news1(out_file1[x],m_list[x])

    #2. 仅给定问询的新闻生成
    gen_news2(out_file2[x],m_list[x])


    
 
    

    
            











