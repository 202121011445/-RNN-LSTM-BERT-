import random
import requests
import json
import time

"""
爬虫模块
"""


def get_comments(url):
    """
    根据指定url获取json格式数据
    :param url: 要爬取的Url
    :return: 返回获取的JSON格式数据
    """
    headers_kv = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers_kv)
        r.raise_for_status()  # 链接不成功时自动抛出异常
        r.encoding = r.apparent_encoding  # 指定编码格式
        s = r.text.replace('fetchJSON_comment98(', '')
        s = s.replace(');', '')
        json_data = json.loads(s)
        return json_data
    except Exception as e:
        print("爬取失败")
        print(e)


def get_good_info(products_id, sku=False, sort_type=5, max_page=1):
    """
    根据指定参数爬取京东平台商品评价信息。为模拟人为操作，每爬取一页评论将耗时3秒
    :parm file_path: csv文件保存路径
    :param products_id: 商品编号列表
    :param sku: 是否仅查询当前商品，默认为False
    :param sort_type: 排序方式。5为默认排序，6为时间排序。默认为5
    :param max_page: 爬取最大页数。默认为1
    """
    # 构造url
    if sku == False:
        url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={0}&score={1}&sortType={2}&page={3}&pageSize=10&isShadowSku=0&fold=1'
    else:
        url = 'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId={0}&score={1}&sortType={2}&page={3}&pageSize=10&isShadowSku=0&fold=1'
    for product_id in products_id:
        bad_nums, good_nums = 0, 0
        for score in [1, 2, 3]:  # score:0为全部评价，1为差评，2为中评，3为好评，5为追加评价
            # 获取商品评论信息,并写入指定文件
            with open('data.csv', 'a', encoding='utf-8') as f:
                loop_num = max_page
                for page in range(0, loop_num):
                    if page > loop_num:
                        break
                    print("商品编号：{0},评价:{1},页数：{2}".format(product_id, score, page))
                    # 获取每页的商品评论
                    spider_url = url.format(product_id, score, sort_type, page)
                    print(spider_url)
                    dic_data = get_comments(spider_url)  # 获取评论
                    # 更新最大爬取页数
                    product_page = dic_data['maxPage']  # 获取评论总页数
                    if product_page < loop_num:
                        loop_num = product_page
                    tem = dic_data['comments']  # 根据key获取value
                    for item in tem:  # 每条评论又分别是一个字典，再继续根据key获取值
                        content = "{},{}".format(1 if item['score'] >= 4 else 0, item['content'])
                        content = content.replace('\n', ' ') + '\n'
                        f.write(content)
                        # 保证好评差评数量相等
                        if score == 1 or score == 2:
                            bad_nums += 1
                        elif score == 3:
                            good_nums += 1
                        if good_nums >= bad_nums:
                            loop_num = 0
                            break
                    time.sleep(random.choice([3, 4, 5, 6]))


if __name__ == '__main__':
    products_id = ["100038004353", "100038004359", "100038004389", "100038004407", "100038089809",
                   "100027789709", "100019791896", "100031406042", "100035225788", "100033901449",
                   "100024857649", "100027028277", "100029707272", "100037199913", "100026819200",
                   "100034042677", "100021889007", "100026330640", "100041054105",
                   "100029049505", "100031192618", "100037323622", "100029467970", "100033725330",
                   "100023070503", "100042265395", "100029666262", "100021631260", "100034036806",
                   "100027685784", "100021172861", "100021468507", "100012462505",
                   "100027700260", "100022883702", "100015039641", "100042392673",
                   "100032670774", "100032670738",
                   "100026761942", "100026761906", "100026761926"]

    lst = get_good_info(products_id, False, 6, 2000)
