# -*- codeing = utif-8 -*-
# @Time：2024/12/1215:07
# @File:promt_chain.py
# @software:PyCharm
#1. 目标抽取
promt_object_extraction="""
你是非常优秀的人工智能助手，提取给定文本中满足用户需求的信息
用户需求：
{query}
需要提取的文本：
{context}
注意事项：
{constraint}
"""
#2. 目标检查
promt_object_check='''
你是非常优秀的人工智能助手，检查给定的文本是否满足用户需求
满足用户需求，则评分score为yes
满足用户部分需求，则评分score为half,给出修正意见
不满足用户需求，则评分score为no,并给出理由
用户需求：
{query}
需要检查的文本：
{context}
注意事项：
1. 如果评分为yes,以{{"score":"yes","reason":"none"}}的json格式返回
2. 如果评分为no,以{{"score":"no","reason":"给出的理由"}}的json格式返回
3. 如果评分为half,以{{"score":"half","reason":"给出的修正意见"}}的json格式返回
'''
#3. 属性提取
promt_attribute_extraction='''
你是非常优秀的信息抽取器,参照示例，从给定文本中完整地提取目标对象的属性信息
目标对象:
{object}
目标属性:
{attribute}
需要提取的文本：
{context}
抽取范例：
{example}
注意事项：
1. 严格遵循抽取范例的格式,以{{"attribute":"抽取的完整属性信息"}}的json格式返回,其中"attribute"是一个固定的字符串
2. 提取的属性信息必须严格遵循文本，不得篡改
{constraint}
'''
#4. 属性检查
promt_attribute_check='''
你是非常优秀的人工智能助手，检查是否从给定文本中正确地提取了目标对象的属性信息
如果正确提取，则评分score为yes
如果提取不正确，则评分score为no,并给出理由
目标对象：
{object}
提取的属性信息
{attribute}
需要检查的文本：
{context}
注意事项：
1. 如果评分为yes,以{{"score":"yes","reason":"none"}}的json格式返回
2. 如果评分为no,以{{"score":"no","reason":"给出的理由"}}的json格式返回
3. 属性信息在目标对象后面
'''

prompt_location = """
你是地理专家，熟悉中国地图，参照给定示例，从给定的query中提取地点信息

用户需求:
{query}

示例:
Example1:
query:撰写今日四川省的气象灾害预警新闻
output:四川省
Example2:
query:撰写今日四川省达州市渠县的气象灾害预警新闻
output:四川省达州市渠县
Example3:
query:撰写今日的气象灾害预警新闻
output:None

注意事项:
如果query中有地点，返回完整的地点信息，如果query中没有地点，返回None

output:
"""

prompt_query="""
你是高级智能助手，参照给定示例，改写query

query:
{query}

Examples:
Example1:
query:陕西省西安市蓝田县气象台发布大风蓝色预警信号
output:中国气象局大风蓝色预警信号
Example2:
query:新疆维吾尔自治区塔城地区托里县气象台发布大风蓝色预警信号
output:中国气象局大风蓝色预警信号
Example3:
query:广东省湛江市气象台发布森林火险橙色预警信号
output:广东省森林火险橙色预警信号
Example4:
query:重庆市忠县气象台发布大雾黄色预警信号
output:重庆市大雾黄色预警信号

注意事项:
1. 如果query中不包含：北京市、重庆市、江苏省、广东省、青海省等特定地点，则统一返回:中国气象局+预警信号;
2. 如果query中包含：北京市、重庆市、江苏省、广东省、青海省等特定地点，则返回：特定地点+预警信号；
3. 输出结果的形式是：地点+预警信号，不要出现其它信息

output:
"""

prompt_query1="""
你是高级智能助手，参照给定示例，改写query

query:
{query}

Examples:
Example1:
query:陕西省西安市蓝田县气象台发布大风蓝色预警信号
output:中国气象局大风蓝色预警信号
Example2:
query:新疆维吾尔自治区塔城地区托里县气象台发布暴雨蓝色预警信号
output:中国气象局暴雨蓝色预警信号
Example3:
query:重庆市忠县气象台发布大雾黄色预警信号
output:中国气象局大雾黄色预警信号

注意事项:
输出结果的形式是：中国气象局+预警信号，不要出现其它信息

output:
"""

prompt_requery="""
你是高级智能助手，参照给定示例，改写query

query:
{query}

Example1:
query:根据中国气象局颁布的政策文件，台风预警信号的分类是什么？
output:中国气象局台风预警信号
query:根据北京市颁布的政策文件，台风预警信号的分类是什么？
output:北京市台风预警信号

Example2:
query:根据青海省颁布的政策文件，大风蓝色预警信号的防御措施是什么？
output:青海省大风蓝色预警信号
query:根据重庆市颁布的政策文件，道路结冰橙色预警信号的防御指南是什么？
output:重庆市道路结冰橙色预警信号

Example3:
query:根据广东省颁布的政策文件，高温黄色预警信号的标准是什么？
output:广东省高温黄色预警信号
query:根据苏州市颁布的政策文件，寒潮蓝色预警信号的标准是什么？
output:苏州市寒潮蓝色预警信号

注意事项:
1. 如果query中不包含：北京市、重庆市、苏州市、广东省、青海省等特定地点，则统一返回:中国气象局+预警信号;
2. 如果query中包含：北京市、重庆市、苏州市、广东省、青海省等特定地点，则返回：特定地点+预警信号；
3. 输出结果的形式是：地点+预警信号，不要出现其它信息

output:
"""

prompt_generation = """
你是气象灾害预警领域的专家，你的任务是根据预警信息与防御指南生成与新闻模板形式一致的气象灾害预警新闻。

预警信息:
{signal}

防御指南:
{defense}

新闻模板:
{example}

注意事项:
1. 生成内容必须基于预警信息和防御指南；
2. 生成内容必须符合新闻模板的形式；
3. 不要产生无关内容；
4. 不要产生虚假信息；
"""

promt_gen_q1='''
你是气象灾害预警领域的专家，你的任务是根据气象信息与政策文件生成气象灾害预警新闻。

气象信息:
{signal}

政策文件:
{policy}


注意事项:
1. 字数不要超过300字；
2. 不要产生虚假信息；
'''

promt_gen_q2='''
你是气象灾害预警领域的专家，你的任务是根据用户问询生成气象灾害预警新闻。

用户问询:
{query}

注意事项:
1. 字数不要超过300字；
2. 不要产生虚假信息；
'''



