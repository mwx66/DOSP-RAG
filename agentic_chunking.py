# -*- codeing = utif-8 -*-
# @Time：2024/12/1814:12
# @File:agentic_chunking.py
# @software:PyCharm
#运用LLM自动抽取目标对象及其属性信息
import re
import os
from args import patterns
from langchain_community.document_loaders import PyPDFLoader
from llm_chain import llm_agent
from promt_chain import promt_object_extraction,promt_object_check,promt_attribute_extraction,promt_attribute_check
from process_data import get_user_input,multi_line_input,dictlist_to_txt,txt_to_dictlist

#1. 遍历指定文件夹下的每个 PDF 文件
def load_pdf(directory):
    """
    参数:
        directory (str): 文件夹路径。
    返回:
        list: [document,...,document]
    """
    documents = []
    for root, _, files in os.walk(directory):  # 遍历目录和子目录
        for file in files:
            if file.lower().endswith('.pdf'):  # 检查文件是否为 PDF
                file_path=os.path.join(root, file)
                loader = PyPDFLoader(file_path)
                document=loader.load()
                documents.append(document)
    return documents

#2. 编写清理pdf页眉页脚页码的函数
def extract_region(metadata):
    # 正则表达式匹配地区名
    pattern = r"data/([^/]+?)\.pdf"
    # 从metadata中的'source'字段提取地区名
    region_match = re.search(pattern, metadata['source'])

    if region_match:
        return region_match.group(1)
    else:
        print('气象灾害预警文件请以发布地区命名，例如北京市：')
        result=input('请输入地区：')
        return result

def clean_pdf(documents):
    pattern = 'none'
    for document in documents:
        metadata=document[0].metadata
        region=extract_region(metadata)
        for key in patterns.keys():
            if key==region:
                pattern=patterns[key]
                break
        if pattern !='none':
            for j in range(len(document)):
                document[j].page_content=re.sub(pattern, "", document[j].page_content)
        else:
            print(f"args.py中不存在{document[0].metadata['source']}的页眉页脚页码正则表达式")
    return documents

#3.编写运用正则表达式精确过滤objects的函数
def objects_filter(objects, document):
    new_objects = []
    seen_objects = set()  # 用于存储已经处理过的对象

    for item in objects:
        index = item['index']
        object_ = item['object']

        # 如果对象未处理且在对应页面内容中，添加到新列表
        if object_ not in seen_objects and object_ in document[index].page_content:
            new_objects.append(item)
            seen_objects.add(object_)  # 将对象标记为已处理

    return new_objects

#4. 批量定制字典的键
def format_unified(attributes, key_mapping):
    """
    :param attributes: list[dict] - 包含字典的列表
    :param key_mapping: dict - 键的映射关系 {旧键: 新键}
    :return: list[dict] - 修改后的字典列表
    """
    if not isinstance(attributes, list) or not all(isinstance(d, dict) for d in attributes):
        raise ValueError("输入必须是字典组成的列表")

    if not isinstance(key_mapping, dict):
        raise ValueError("键映射关系必须是字典")

    new_attributes = []
    for original_dict in attributes:
        modified_dict = {}
        for key, value in original_dict.items():
            new_key = key_mapping.get(key, key)  # 如果找不到对应的新键，则保留旧键
            modified_dict[new_key] = value
        new_attributes.append(modified_dict)

    return new_attributes


#5. 编写获取目标对象及其索引的函数
def object_index(document, self_maxnum, extraction_api, check_api,query,constraint,sub_query):
    '''
    :param document: pdf document 对象
    :param self_maxnum: 自反思阈值
    :param extraction_api: 抽取的api设置
    :param check_api: 检查的api设置
    :param query:目标信息抽取的具体需求
    :param constraint:目标信息抽取的限制条件
    :param sub_query：目标信息检查的要求与示例
    :return: List of dictionaries containing objects and their indices.
    '''
    def perform_extraction(query, context, constraint, api):
        return llm_agent(promt_object_extraction, {
            'query': query,
            'context': context,
            'constraint': constraint
        }, api)

    def perform_check(query, sub_context, api):
        return llm_agent(promt_object_check, {
            'query': query,
            'context': sub_context
        }, api)

    print("*" * 50 + "目标抽取开始" + "*" * 50)
    objects = []

    for index, page in enumerate(document):
        print(f"{'*' * 50}第{index + 1}页{'*' * 50}")
        context = page.page_content
        #第一次提取
        extractions = perform_extraction(query, context, constraint, extraction_api)
        print('extractions:', extractions)

        if 'None' in extractions.values():
            continue
        #有限次反思与提取
        for attempt in range(self_maxnum):
            sub_context = str(extractions)
            checks = perform_check(sub_query, sub_context, check_api)
            print('checks:', checks)

            if checks.get('score') == 'yes':
                objects.extend({
                    'object': value,
                    'index': index
                } for value in extractions.values())
                break
            else:
                new_constraint = constraint + '\n' + checks.get('reason', '')
                extractions = perform_extraction(query, context, new_constraint, extraction_api)
                print('extractions:', extractions)

                if 'None' in extractions.values():
                    break
        else:
            #人工干预
            while True:
                print("需要人为干预给出修正意见：输入'end' 结束输入：")
                modification = multi_line_input()

                new_constraint = constraint + '\n' + modification
                extractions = perform_extraction(query, context, new_constraint, extraction_api)
                print('extractions:', extractions)

                if 'None' in extractions.values():
                    break

                print("检查抽取是否正确，正确输入'1',错误输入'0',输入'end' 结束输入")
                check = multi_line_input()

                if check == '1':
                    objects.extend({
                        'object': value,
                        'index': index
                    } for value in extractions.values())
                    break

    return objects

#6. 编写object的attribute抽取与核查函数
def extract_context(document, index, next_index=None):
    """Extract the full context based on index range."""
    context = document[index].page_content
    if next_index is not None and index != next_index:
        for j in range(1, next_index - index + 1):
            context += document[index + j].page_content
    elif next_index is None:
        for j in range(1, len(document) - index):
            context += document[index + j].page_content
    return context

def object_attribute(document, objects, self_maxnum, extraction_api, check_api, attribute, constraint, example):
    metadata = document[0].metadata
    region = extract_region(metadata)
    attributes = []

    print("*" * 50 + "属性抽取开始" + "*" * 50)

    for i, obj in enumerate(objects):
        object_name = obj['object']
        index = obj['index']
        next_index = objects[i + 1]['index'] if i + 1 < len(objects) else None

        print("*" * 50 + f"{object_name}" + "*" * 50)
        context = extract_context(document, index, next_index)
        print('context:', context)

        input_data = {
            'object': object_name,
            'attribute': attribute,
            'context': context,
            'constraint': constraint,
            'example': example
        }
        extraction = llm_agent(promt_attribute_extraction, input_data, extraction_api)
        print('*' * 200)
        print('extraction:', extraction)

        check_data = {
            'object': object_name,
            'attribute': extraction['attribute'],
            'context': context
        }
        check = llm_agent(promt_attribute_check, check_data, check_api)
        print('*' * 200)
        print('check:', check)

        if check.get('score') == 'yes':
            attributes.append({
                'region': region,
                'object': object_name,
                'attribute': extraction['attribute'],
            })
            continue

        # Retry logic for self-correction
        for attempt in range(self_maxnum):
            input_data['constraint'] = constraint+'\n' + check.get('reason', '')
            extraction = llm_agent(promt_attribute_extraction, input_data, extraction_api)
            print('*' * 200)
            print('extraction:', extraction)

            check_data['attribute'] = extraction['attribute']
            check = llm_agent(promt_attribute_check, check_data, check_api)
            print('*' * 200)
            print('check:', check)

            if check.get('score') == 'yes':
                attributes.append({
                    'region': region,
                    'object': object_name,
                    'attribute': extraction['attribute'],
                })
                break
        else:
            while True:
                print("需要人为干预给出修正意见：输入'end' 结束：")
                modification = multi_line_input()
                input_data['constraint'] =constraint+ '\n' + modification
                extraction = llm_agent(promt_attribute_extraction, input_data, extraction_api)
                print('*' * 200)
                print('extraction:', extraction)

                print("检查抽取是否正确，正确输入'1',错误输入'0',输入'end' 结束输入")
                check_input = multi_line_input()
                if check_input == '1':
                    attributes.append({
                        'region': region,
                        'object': object_name,
                        'attribute': extraction['attribute'],
                    })
                    break
    return attributes

#7.抽取给定目录下所有pdf的目标对象及其所在页码
def object_index_extraction(test_doc,self_num,save_path,e_api,c_api,query,constraint,sub_query):
    '''
    抽取给定pdf的目标对象及其所在页码
    '''
    metadata = test_doc[0].metadata
    region = extract_region(metadata)
    print("#"*50+f"{region}.pdf"+"#"*50)
    object_indexs=object_index(test_doc,self_num,e_api,c_api,query,constraint,sub_query)
    new_object_indexs=objects_filter(object_indexs,test_doc)
    print('目标对象个数：',len(new_object_indexs))
    dictlist_to_txt(save_path, new_object_indexs)#file_path='data/object_indexs.txt'
    print(f"针对文件{region}.pdf,LLM提取的目标对象已经保存到{save_path}")

#8.抽取给定目录下所有pdf的目标对象的属性信息
def object_attribute_extraction(test_doc,index_path,self_num,save_path,key_mapping,e_api,c_api,attribute, constraint, example):

    metadata = test_doc[0].metadata
    region = extract_region(metadata)
    print("#" * 50 + f"{region}.pdf" + "#" * 50)
    object_index = txt_to_dictlist(index_path)
    object_attributes = object_attribute(test_doc,object_index, self_num,e_api,c_api,attribute, constraint, example)
    new_object_attribute = format_unified(object_attributes, key_mapping)
    dictlist_to_txt(save_path, new_object_attribute)
    print(f"针对文件{region}.pdf,LLM提取的目标对象的属性信息已经保存到{save_path}")


if __name__ == "__main__":
    #1 参数设定
    query = get_user_input("请输入目标信息抽取的具体需求：")
    constraint1 = get_user_input("请输入目标信息抽取的限制条件：")
    sub_query = get_user_input("请输入目标信息检查的要求与示例：")
    attribute=get_user_input("请输入目标信息抽取的具体属性:")
    constraint2=get_user_input("请输入属性信息抽取的限制条件:")
    example=get_user_input("请输入属性信息抽取的范例，以{'attribute':'抽取的属性信息'}格式组织范例")
    pdf_path = 'data/'
    documents = load_pdf(pdf_path)
    new_documents = clean_pdf(documents)
    self_num = 2
    e_api=['ZZZ_KEY', 'ZZZ_URL', 'ZZZ_MODEL_2']
    c_api=['ZZZ_KEY', 'ZZZ_URL', 'ZZZ_MODEL_2']
    o_index_list=[]
    key_mapping = {
        'region': '地区',
        'object': '预警信号',
        'attribute': '分类'
    }#key_mapping = {'region': '地区','object': '预警信号','attribute': '标准'/'防御措施'}
    #2. 目标抽取
    for doc in new_documents:
        o_save_path = get_user_input("请输入目标索引的保存路径:")#'data/青海省_object1_index.txt'
        o_index_list.append(o_save_path)
        object_index_extraction(doc, self_num, o_save_path, e_api, c_api,query,constraint1,sub_query)
    #3. 属性抽取
    for i in range(len(new_documents)):
        a_save_path = get_user_input("请输入提取信息的保存路径:")#'data/c_chunks_llm.txt' or 'data/d_chunks_llm.txt' or 'data/e_chunks_llm.txt'
        doc=new_documents[i]
        index_path=o_index_list[i]
        object_attribute_extraction(doc, index_path, self_num, a_save_path, key_mapping, e_api, c_api,attribute,constraint2,example)





















