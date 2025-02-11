# -*- codeing = utif-8 -*-
# @Time：2024/12/212:53
# @File:util.py
# @software:PyCharm
#气象预警新闻生成通道
import re
import os
import datetime
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By  # 导入 By 模块
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from promt_chain import prompt_location,prompt_query,prompt_generation
from llm_chain import llm_agent_s
from e_retriever import hybrid_retriver2
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from process_data import get_user_input

#1. 定向的网络搜索，获取实时信息
def fetch_signal_info(url):
    """
    从 URL 提取预警信号内容
     response = requests.get(url)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    signal_info = soup.find(id='alarmtext').get_text().strip()
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查 HTTP 请求是否成功
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找元素，增加容错机制
        element = soup.find(id='alarmtext')
        if element is None:
            raise ValueError("未找到 id='alarmtext' 的元素，请检查页面结构或 HTML 内容。")

        signal_info = element.get_text().strip()
        return signal_info

    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

def get_next_page(driver, button):
    """
    获取下一页的按钮
    """
    try:
        pagination = driver.find_element(By.CSS_SELECTOR, "#M-box3")
        next_page = pagination.find_element(By.LINK_TEXT, str(button))
        return next_page
    except NoSuchElementException:
        print("未找到下一页按钮")
        return None

def get_dynamic_info(request_time, pre_info):
    '''
    从中央气象台获取此时的天气预警信息
    '''
    # 初始化 Selenium WebDriver
    service = ChromeService(executable_path="driver/chromedriver-win64/chromedriver.exe")
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        if request_time == 1:
            # 初始化
            need_time = datetime.datetime.now().strftime('%Y/%m/%d')
            flag = True
            button = 1
            dynamic_info = []

            # 打开中央气象台首页
            driver.get('http://www.nmc.cn/publish/alarm.html')
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "alarm-item"))
            )

            # 提取预警信号标题与信息
            while flag:
                try:
                    even_elements = driver.find_elements(By.CLASS_NAME, "even.alarm-item")
                    odd_elements = driver.find_elements(By.CLASS_NAME, "odd.alarm-item")
                    # 检查元素是否为空
                    if not even_elements and not odd_elements:
                        print("未找到任何预警元素，结束循环")
                        break

                    elements = even_elements + odd_elements

                    try:
                        min_time = odd_elements[-1].find_element(By.CLASS_NAME, 'date').text.split()[0]
                    except IndexError:
                        print("找不到日期，可能预警列表为空")
                        break

                    if min_time == need_time:
                        for ele in elements:
                            title = ele.find_element(By.TAG_NAME, 'a').text
                            url = ele.find_element(By.TAG_NAME, 'a').get_attribute('href')
                            signal_info = fetch_signal_info(url)
                            dynamic_info.append({'title': title, 'signal_info': signal_info, 'url': url})
                        button += 1
                        next_page = get_next_page(driver, button)
                        if next_page:
                            next_page.click()
                            time.sleep(2)  # 等待新页面加载
                        else:
                            break
                    else:
                        for ele in elements:
                            news_time = ele.find_element(By.CLASS_NAME, 'date').text.split()[0]
                            if news_time == need_time:
                                title = ele.find_element(By.TAG_NAME, 'a').text
                                url = ele.find_element(By.TAG_NAME, 'a').get_attribute('href')
                                signal_info = fetch_signal_info(url)
                                dynamic_info.append({'title': title, 'signal_info': signal_info, 'url': url})
                            else:
                                flag = False
                except StaleElementReferenceException:
                    print("元素失效，重新加载当前页面")
                    driver.refresh()
                    time.sleep(2)

        else:
            # 初始化
            need_info = pre_info[0]['title']
            flag = True
            button = 1
            dynamic_info = []

            # 打开中央气象台首页
            driver.get('http://www.nmc.cn/publish/alarm.html')
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "alarm-item"))
            )

            # 提取预警信号标题与信息
            while flag:
                try:
                    even_elements = driver.find_elements(By.CLASS_NAME, "even.alarm-item")
                    odd_elements = driver.find_elements(By.CLASS_NAME, "odd.alarm-item")
                    elements = even_elements + odd_elements

                    titles = [ele.find_element(By.TAG_NAME, 'a').text for ele in elements]

                    if need_info in titles:
                        for ele in elements:
                            title = ele.find_element(By.TAG_NAME, 'a').text
                            if title == need_info:
                                flag = False
                                break
                            url = ele.find_element(By.TAG_NAME, 'a').get_attribute('href')
                            signal_info = fetch_signal_info(url)
                            dynamic_info.append({'title': title, 'signal_info': signal_info, 'url': url})
                    else:
                        button += 1
                        next_page = get_next_page(driver, button)
                        if next_page:
                            next_page.click()
                            time.sleep(2)  # 等待新页面加载
                        else:
                            break
                except StaleElementReferenceException:
                    print("元素失效，重新加载当前页面")
                    driver.refresh()
                    time.sleep(2)

    finally:
        driver.quit()  # 确保释放资源

    return dynamic_info

#2.用户交互
def get_boolean_input():
    while True:
        user_input = input("请输入True或False,True表示继续提问，False表示退出: ").strip()

        # 判断用户输入是否是'True'或'False'
        if user_input.lower() == 'true':
            return True
        elif user_input.lower() == 'false':
            return False
        else:
            print("输入无效，请输入True或False。")

def get_number_input(max_value):
    while True:
        user_input = input(f"需要撰写几篇气象灾害预警新闻,请输入一个不超过{max_value}的数字: ")
        # 检查输入是否为数字
        if user_input.isdigit():
            num = int(user_input)

            # 检查数字是否小于或等于最大值
            if num <= max_value:
                return num
            else:
                print(f"输入的数字超过了最大值{max_value}，请重新输入。")
        else:
            print("输入无效，请输入一个数字。")


#3. 静态信息加载+气象灾害预警新闻生成
def weather_rag_news(subquery,dynamic_info,vector_db):
    #输入获取
    defense = hybrid_retriver2(subquery,vector_db,5)
    print('defense:','\n',defense)
    signal = dynamic_info['title'] + '\n' + dynamic_info['signal_info']
    print('signal:','\n',signal)
    example =get_user_input('请输入气象灾害预警新闻示例，选择默认模板请输入<None>：')
    if example.lower() == 'none':  # 允许用户输入大小写混合
        try:
            with open('answer/example2.txt', "r", encoding="utf-8") as f:
                example = f.read()
            print('已加载默认模板内容。')
        except FileNotFoundError:
            print('错误：默认模板文件未找到，请检查文件路径。')
            example = ''  # 如果加载失败，可以初始化为空字符串或提供其他备选方案
        except Exception as e:
            print(f'发生未知错误：{e}')
            example = ''
    else:
        print('用户提供的新闻示例：', example)
    #LLM配置
    promt_=prompt_generation
    input_={
              "signal": signal,
              "defense": defense,
              "example": example,
            }

    response=llm_agent_s(promt_,input_,'ZZZ_MODEL_G')

    return response

if __name__ == "__main__":
    db_path = 'data/index_db/Jinaai/chunk_d_index'
    embeddings = HuggingFaceBgeEmbeddings(model_name='jinaai/jina-embeddings-v2-base-zh')
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    flag = True
    request_num = 0
    dynamic_info = []
    api_='ZZZ_MODEL_1'
    #撰写某个地区的气象灾害预警新闻
    while flag:
        request_num += 1
        query=get_user_input('请输出你的请求,需要包含地点：')
        input_1={'query':query}
        location = llm_agent_s(prompt_location,input_1,api_)
        print('用户请求中的location为:', location)
        if location.lower() == 'none':
            print('query中未给出地点信息,请重新定义query或者退出')
            request_num = 0
            flag = get_boolean_input()
        else:
            # 获取此时的气候预警信息
            current_time = datetime.datetime.now()
            dynamic_info = get_dynamic_info(request_num, dynamic_info) + dynamic_info
            info_nums = len(dynamic_info)
            print(f"截止{current_time},从中央气象局共获取{info_nums}条气象灾害预警信息，前20条信息详情如下：")
            for info in dynamic_info[:20]:
                print(info)
            locations = []
            for item in dynamic_info:
                title = item['title']
                match = re.search(r"(.+?)气象台", title)
                if match:
                    locations.append(match.group(1))
            relevant_info = []
            for i in range(len(locations)):
                if location in locations[i]:
                    relevant_info.append(dynamic_info[i])
            if len(relevant_info) == 0:
                print(f"目前{location}未发生气象灾害")
                flag = get_boolean_input()
            else:
                print(f"目前{location}的气象灾害预警信息有{len(relevant_info)}条，详情如下：")
                for info in relevant_info:
                    print(info)
                news_k = get_number_input(len(relevant_info))
                for info in relevant_info[:news_k]:
                    original_query = info['title']
                    input_2={'query':original_query}
                    sub_query = llm_agent_s(prompt_query,input_2,api_)
                    print('sub_query:', sub_query)
                    # 检索增强生成
                    news = weather_rag_news(sub_query, info, db)
                    print('*' * 50)
                    print('News:', news)
                    save_path = os.path.join('answer', 'news.txt')
                    with open(save_path, 'a') as file:
                        file.write(news + '\n')  # 使用换行符分割每条文本
                flag = get_boolean_input()
