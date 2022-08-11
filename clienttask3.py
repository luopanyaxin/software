#!coding=utf-8

import socket
import os
import sys
import struct
import pandas as pd
import json

def socket_client():
    ## 开始链接，127.0.0.1，9001表示本机的端口这里不用修改
    try:
        s = socket.socket()
        s.connect(('127.0.0.1', 9001))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    ## 服务器第一次回信，会收到welcome代表连上
    print(s.recv(1024))
    ## 给边缘设备发送的内容：epoch数、learning_rate大小，以及从excel读取的data数据，存放在send_dic字典中（可添加新的内容：send_dic['xxx']=xxx）
    param = pd.read_excel('param.xlsx')
    epoch=20
    learning_rate = 0.01
    param['迭代次数'][0] = epoch
    param['学习率'][0] = learning_rate
    df = pd.read_excel('测试数据集/测试数据集/任务三data.xlsx')
    id = 1000000001

    data = []
    true_data = []
    for i in range(len(df['用户编号'])):
        if i != len(df['用户编号']) - 1:
            if df['用户编号'][i] == id:
                true_data.append(int(df['缴费金额（元）'][i]))
            else:
                id = df['用户编号'][i]
                data.append(true_data)
                true_data = []
                true_data.append(int(df['缴费金额（元）'][i]))
        else:
            true_data.append(int(df['缴费金额（元）'][i]))
            data.append(true_data)

    send_dic = {}
    send_dic['data']=data
    send_dic['learning_rate']=learning_rate
    send_dic['epoch']=epoch
    ## 将字典打包为字符串发过去
    json_string = json.dumps(send_dic)
    s.send(json_string.encode())


    ## 接受边缘设备的输出结果：画图所需的10个用户的损失（result['loss']），训练时间(result['spend_time'])，以及预测值(result['pred'],这个就是你们之前写的存放在total数组中的数据)
    ## 8192表示一次可以接受到最大的byte数为8192，长度太长会被截断，可以调整
    result = json.loads(s.recv(8192))
    s.close()
    return result




if __name__ == '__main__':
    result = socket_client()
    print(result)

