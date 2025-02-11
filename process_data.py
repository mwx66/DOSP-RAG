# -*- codeing = utif-8 -*-
# @Time：2024/12/270:01
# @File:process_data.py
# @software:PyCharm
#数据加载、清洗、格式转换
import os
import json
from langchain.schema import Document
import re
import random

#定义多行输入函数，以便输入用户需求和限制
def multi_line_input():
    lines = []
    while True:
        line = input()  # 获取一行输入
        if line == "end":  # 如果输入 'end'，结束输入
            break
        lines.append(line)  # 将输入的每一行添加到列表
    # 将所有行连接为一个字符串，行与行之间用换行符分隔
    user_input = "\n".join(lines)
    return user_input

def get_user_input(prompt):
    print(prompt + " 输入'end' 结束：")
    return multi_line_input()

#去掉文本中的换行符
def remove_newlines(data):
    """
    Remove newline characters from the input, supporting strings, lists of dictionaries, and document objects.

    Args:
        data: Input data, can be a string, list of dictionaries, or a document-like object.

    Returns:
        Processed data with newline characters removed.
    """
    if isinstance(data, str):  # If the input is a string
        return data.replace('\n', '').replace('\r', '')

    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):  # If it's a list of dictionaries
        for dictionary in data:
            for key, value in dictionary.items():
                if isinstance(value, str):
                    dictionary[key] = value.replace('\n', '').replace('\r', '')
        return data

    elif hasattr(data, 'page_content') and isinstance(data.page_content, str):  # If it's a document-like object
        data.page_content = data.page_content.replace('\n', '').replace('\r', '')
        return data

    elif isinstance(data, list) and all(hasattr(d, 'page_content') for d in data):  # List of document-like objects
        for document in data:
            if hasattr(document, 'page_content') and isinstance(document.page_content, str):
                document.page_content = document.page_content.replace('\n', '').replace('\r', '')
        return data

    else:
        raise TypeError(
            "Unsupported data type. Supported types: string, list of dictionaries, or document-like objects.")

#去除中文数字
def remove_chinese_numbers(dict_list):

    chinese_number_pattern =re.compile(r"[(（]?[一二三四五六七八九十百千万零]+[)）]?[、]?") #re.compile(r"[(（]?[一二三四五六七八九十百千万零]+[)）]?")

    # 遍历列表中的每个字典
    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, str):  # 仅对字符串类型的值处理
                d[key] = chinese_number_pattern.sub("", value)  # 去除中文数字
            if isinstance(value, list):
                d[key]=[chinese_number_pattern.sub("", item) for item in value]
    return dict_list

#6.将列表转化为txt
def dictlist_to_txt(file_path, dict_list):
    """
    将包含字典的列表保存到一个txt文件中。
    :param file_path: str - 保存的文件路径。
    :param dict_list: List[dict] - 要保存的字典列表。
    """
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, 'a', encoding='utf-8') as file:
        for d in dict_list:
            file.write(f"{d}\n")

#7. 将txt转化为列表
def txt_to_dictlist(file_path):
    """
    从txt文件加载包含字典的列表。

    :param file_path: str - 加载的文件路径。
    :return: List[dict] - 从文件中加载的字典列表。
    """
    dict_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            dict_list.append(eval(line.strip()))
    return dict_list

#字典转json文件
def dict_to_json(dictionary, file_path):
    """将字典转换为 JSON 文档并保存到指定路径"""
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
        print(f"字典已成功转换为 JSON 并保存到 {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")
#将字典中的值融合，转为一个json文件
def valuef_json(file_path,save_path):
    dict_={}
    list_=txt_to_dictlist(file_path)
    for i in range(len(list_)):
        value = list(list_[i].values())
        for j in range(len(value)):
            if isinstance(value[j], list):
                value[j]=' '.join(value[j]) # 使用空格分隔
        value=' '.join(value)
        key = str(i + 1)
        dict_[key] = [value]
    dict_to_json(dict_, save_path)#'data/defense_pre.json'
#将字典列表中特殊的键对应的值取出来，转为json文件
def valuek_json(file_path,save_path,key):
    dict_ = {}
    list_ = txt_to_dictlist(file_path)
    for i in range(len(list_)):
        item=list_[i]
        if isinstance(item.get(key), str):
            k = str(i + 1)
            dict_[k] = [item.get(key)]
        if isinstance(item.get(key), list):
            value= ','.join(item.get(key))
            k = str(i + 1)
            dict_[k] = [value]
    dict_to_json(dict_, save_path)

def get_qa_data(file_path,save_path):
    '''
    条件：字典列表中前2个key的值+最后一个key组成问询，最后一个key的value为answer
    :param file_path:
    :param save_path:
    :return:
    '''
    dict_list=txt_to_dictlist(file_path)
    qa_list=[]
    for dict in dict_list:
        doc={}
        k_list=list(dict.keys())
        v_list=list(dict.values())
        source=v_list[0]
        object=v_list[1]
        attribute=k_list[2]
        q=f"根据{source}颁布的政策文件,{object}的{attribute}是什么？"
        a=f"根据{source}颁布的政策文件,{object}的{attribute}为：{v_list[2]}"
        doc['query']=q
        doc['answer']=a
        qa_list.append(doc)
    dictlist_to_txt(save_path,qa_list)
    return qa_list

def get_source(file_path,save_path):
    dict_list = txt_to_dictlist(file_path)
    s_list=[]
    for dict in dict_list:
        doc={}
        v_list = list(dict.values())
        s=v_list[0]
        doc['source']=s
        s_list.append(doc)
    dictlist_to_txt(save_path, s_list)
    return s_list

def documents_to_txt(documents, file_path):
    """
    将一个包含多个 document 对象的列表转换为文本文件。
    :param documents: 包含多个 document 对象的列表
    :param file_path: 保存为 txt 文件的路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for idx, document in enumerate(documents):
                file.write(f"Document {idx + 1}:\n")  # 写入文档编号

                # 写入元数据
                if hasattr(document, 'metadata') and isinstance(document.metadata, dict):
                    file.write("Metadata:\n")
                    for key, value in document.metadata.items():
                        file.write(f"{key}: {value}\n")
                else:
                    file.write("Metadata:\n[No Metadata Found]\n")

                file.write("\n")  # 添加空行分隔

                # 写入内容
                if hasattr(document, 'page_content') and isinstance(document.page_content, str):
                    file.write("Content:\n")
                    file.write(document.page_content)
                else:
                    file.write("Content:\n[No Content Found]")

                file.write("\n" + "-" * 50 + "\n")  # 添加分隔线
        print(f"Documents successfully saved to {file_path}.")
    except Exception as e:
        print(f"Error saving documents to txt: {e}")

def text_to_dict(raw_text):
    """
    Parses a raw text string into a dictionary format.

    Args:
        raw_text (str): The input text in string format.

    Returns:
        dict: A dictionary containing extracted fields.
    """
    # Remove unwanted characters and normalize the input
    patterns = {
        "地区": r"地区:\s*([^\n]+)",
        "预警信号": r"预警信号:\s*([^\n]+)",
        "防御指南": r"防御指南:\s*(.+)"
    }
    # Extract fields using the patterns
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, raw_text)
        if match:
            data[key] = match.group(1).strip()

    return data

def text_to_document(raw_text):
    """
    Transforms raw text into a Document object with metadata.

    Args:
        raw_text (str): The input text containing warning signal information.

    Returns:
        Document: A Document object with metadata for the region.
    """
    # Parse the text into a dictionary
    data = text_to_dict(raw_text)
    print(data)

    # Extract metadata and content
    region = data.get("地区", "未知")
    content = raw_text

    # Create Document object
    document = Document(
        page_content=content,
        metadata={"地区": region}
    )

    return document

def get_txt_path(directory_path):
    """
    Get paths of all .txt files from a given directory.

    Args:
        directory_path (str): The path to the directory containing .txt files.

    Returns:
        list: A list of full paths to .txt files in the directory.
    """
    txt_file_paths = []
    # Check if the provided path is a directory
    if not os.path.isdir(directory_path):
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            txt_file_paths.append(file_path)

    return txt_file_paths


def textlist_to_json(output_file,string_list):
    if not isinstance(string_list, list) or not all(isinstance(item, str) for item in string_list):
        raise ValueError("Input must be a list of strings.")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create a dictionary with numbered keys starting from 1
    result = {str(idx + 1): [string.strip()] for idx, string in enumerate(string_list)}

    # Save the dictionary to a JSON file
    with open(output_file, 'a', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
    print(f"JSON file saved to {output_file}")

def gen_qkn(filepath):
    c_data = txt_to_dictlist(filepath)
    nums = [89, 229, 449, 813, 1162, 1661, 1933, 2228, 2413, 2554]
    samples = []
    for i in range(len(nums)):
        if i == 0:
            sample = random.sample(c_data[:nums[i]], 10)
            print(sample[0]['date'])
            samples.extend(sample)
        else:
            sample = random.sample(c_data[nums[i - 1]:nums[i]], 10)
            print(sample[0]['date'])
            samples.extend(sample)
    query = []
    dk = []
    news= []
    for i in range(len(samples)):
        q='撰写'+samples[i]['date']+samples[i]['query']+'的气象灾害预警新闻'
        query.append({str(i+1):q})
        dk.append({str(i+1):samples[i]['dynamic_info']})
        news.append(samples[i]['query']+'\n'+samples[i]['news'])
    dictlist_to_txt('data/gens_e/query.txt', query)
    print(f"query已经成功保存至data/gens_e/query.txt")
    dictlist_to_txt('data/gens_e/dk.txt', dk)
    print(f"dk已经成功保存至data/gens_e/dk.txt")
    textlist_to_json('data/gens_e/goldnews_zh.json',news)

if __name__ == "__main__":

    '''   
    #1. 合成气象灾害领域的问答数据集  
    f_p=['data/c_chunks_llm.txt','data/d_chunks_llm.txt','data/e_chunks_llm.txt']
    s_p=['data/c_qa.txt','data/d_qa.txt','data/e_qa.txt']
    for i in range(len(f_p)):
        fp=f_p[i]
        sp=s_p[i]
        get_qa_data(fp, sp)
        
    #2. 得到各文件的来源
    f_p = ['data/c_chunks_llm.txt', 'data/d_chunks_llm.txt', 'data/e_chunks_llm.txt']
    s_p = ['data/c_source.txt', 'data/d_source.txt', 'data/e_source.txt']
    for i in range(len(f_p)):
        fp = f_p[i]
        sp = s_p[i]
        get_source(fp, sp)

    #3. 将文本转化为document对象，并加上地区作为元数据
    raw_text = "地区: 中国气象局\n预警信号: 台风蓝色预警信号\n防御指南: 1.政府及相关部门按照职责做好防台风准备工作；2.停止露天集体活动和高空等户外危险作业；3.相关水域水上作业和过往船舶采取积极的应对措施，如回港避风或者绕道航行等；4.加固门窗、围板、棚架、广告牌等易被风吹动的搭建物，切断危险的室外电源。"
    document = text_to_document(raw_text)
    print(document)
    
    #4.构建生成质量评估的问询数据、动态信息数据以及真实新闻
    date = [
        '2025年1月1日',
        '2024年1月2日',
        '2024年12月24日',
        '2024年12月25日',
        '2024年12月26日',
        '2024年12月27日',
        '2024年12月28日',
        '2024年12月29日',
        '2024年12月30日',
        '2024年12月31日',
    ]
    path1 = get_txt_path('data/gens_e/gold_news')
    path2 = get_txt_path('data/gens_e/half_news')
    data = []
    for i in range(len(path1)):
        list1 = txt_to_dictlist(path1[i])
        list2 = txt_to_dictlist(path2[i])
        for j in range(len(list1)):
            list2[j]['news'] = list1[j]['content']
            list2[j]['date']= date[i]
        data.extend(list2)
    dictlist_to_txt('data/gens_e/c_data.txt', data)
    
    filepath='data/gens_e/c_data.txt'
    gen_qkn(filepath)
    '''













