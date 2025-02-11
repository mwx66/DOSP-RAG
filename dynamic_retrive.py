# -*- codeing = utif-8 -*-
# @Time：2024/12/2421:41
# @File:dynamic_retrive.py
# @software:PyCharm
#.从中央气象台爬取每日的气象预警新闻与气象预警信息，并存到txt中，作为评估数据
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from process_data import dictlist_to_txt

def fetch_signal_info(url):
    """
    从 URL 提取所有 id 为 'alarmtext' 的预警信号内容，并将其转化为一个字符串
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查 HTTP 请求是否成功
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找所有符合 id='alarmtext' 的元素
        elements = soup.find_all(id='alarmtext')
        if not elements:
            raise ValueError("未找到任何 id='alarmtext' 的元素，请检查页面结构或 HTML 内容。")

        # 提取并清理每个元素的文本
        signal_info_list = [element.get_text().strip() for element in elements]
        signal_info=signal_info_list[0]

        # 将列表中的内容转化为字符串，使用换行符连接
        signal_defense_info = "\n".join(signal_info_list)
        return signal_info,signal_defense_info

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

def get_double_infos(URL,need_time):
    '''
    从中央气象台获取当天的气象灾害预警新闻
    '''
    # 初始化 Selenium WebDriver
    service = ChromeService(executable_path="driver/chromedriver-win64/chromedriver.exe")
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    flag = True
    button = 1
    #need_time = datetime.datetime.now().strftime('%Y/%m/%d')
    gold_news = []
    half_news=[]

    # 打开中央气象台首页
    driver.get(URL)#'http://www.nmc.cn/publish/alarm.html'
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "alarm-item"))
    )
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
                    signal_info,sigal_defense_info = fetch_signal_info(url)
                    half_news.append({'query': title, 'dynamic_info': signal_info})
                    gold_news.append({'title': title, 'content': sigal_defense_info})
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
                        signal_info, sigal_defense_info = fetch_signal_info(url)
                        half_news.append({'query': title, 'dynamic_info': signal_info})
                        gold_news.append({'title': title, 'content': sigal_defense_info})
                    else:
                        flag = False
        except StaleElementReferenceException:
            print("元素失效，重新加载当前页面")
            driver.refresh()
            time.sleep(2)
    driver.quit()  # 确保释放资源
    return gold_news,half_news

if __name__ == "__main__":
    need_time = '2024/12/28'#datetime.datetime.now().strftime('%Y/%m/%d')
    url='http://www.nmc.cn/publish/alarm.html'
    gold_news,half_news = get_double_infos(url,need_time)
    file_path1 = 'data/gens_e/gold_news_1228.txt'
    dictlist_to_txt(file_path1, gold_news)
    file_path2='data/gens_e/half_news_1228.txt'
    dictlist_to_txt(file_path2, half_news)


