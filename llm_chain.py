# -*- codeing = utif-8 -*-
# @Time：2024/12/1715:03
# @File:llm_chain.py
# @software:PyCharm
import os
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.output_parser import StrOutputParser

def llm_agent(promt_,input_,api_):
    '''
    :param promt_: 提示模板,str
    :param input_: 输入变量,dict,{'query':query,'context':context}
    :param api_: API设置,list, ['ZZZ_KEY','ZZZ_URL','ZZZ_MODEL']
    :return:
    '''
    # 加载环境变量
    load_dotenv()

    os.environ["LANGCHAIN_API_KEY"] = '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'
    os.environ["LANGCHAIN_TRACING_V2"] = "false"#停止运用langsmith追踪
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "langchain_for_llm_application_development"

    API_SECRET_KEY = os.environ.get(api_[0])
    BASE_URL = os.environ.get(api_[1])
    Model = os.environ.get(api_[2])

    # 初始化 LLM 模型
    llm = ChatOpenAI(
        model=Model,
        base_url=BASE_URL,
        api_key=API_SECRET_KEY,
        temperature=0,
    )

    # 创建 Prompt 模板
    prompt = PromptTemplate(template=promt_,input_variables=input_.keys())

    # 构建链条
    chain = prompt | llm | JsonOutputParser()

    try:
        # 调用链条
        response = chain.invoke(input_)

        # 如果 response 是字典，直接返回
        if isinstance(response, dict):
            return response

        # 如果是字符串类型的响应，进行替换并解析
        if isinstance(response, str):
            response = response.replace("'", '"')  # 替换单引号为双引号
            response_json = json.loads(response)  # 解析为 JSON 格式
            return response_json

        # 如果响应格式不对，抛出错误
        raise ValueError("无法处理的响应格式")

    except Exception as e:
        # 如果出现任何错误，返回一个错误消息
        print(f"错误: {e}")
        return {"error": f"An error occurred: {str(e)}"}

def llm_agent_s(promt_,input_,model):
    # 加载环境变量
    load_dotenv()

    os.environ["LANGCHAIN_API_KEY"] = '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'
    os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 停止运用langsmith追踪
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "langchain_for_llm_application_development"

    API_SECRET_KEY = os.environ.get('ZZZ_KEY')
    BASE_URL = os.environ.get('ZZZ_URL')
    Model = os.environ.get(model)

    # 初始化 LLM 模型
    llm = ChatOpenAI(
        model=Model,
        base_url=BASE_URL,
        api_key=API_SECRET_KEY,
        temperature=0,
    )

    # 创建 Prompt 模板
    prompt = PromptTemplate(template=promt_, input_variables=input_.keys())

    # 构建链条
    chain = prompt | llm | StrOutputParser()

    try:
        # 调用链条
        response = chain.invoke(input_)

        # 如果 response 是字典，直接返回
        if isinstance(response, str):
            return response
        # 如果响应格式不对，抛出错误
        raise ValueError("无法处理的响应格式")

    except Exception as e:
        # 如果出现任何错误，返回一个错误消息
        print(f"错误: {e}")
        return {"error": f"An error occurred: {str(e)}"}






