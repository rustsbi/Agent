
import os
import time
def truncate_filename(filename, max_length=200):
    # 获取文件名后缀
    file_ext = os.path.splitext(filename)[1]

    # 获取不带后缀的文件名
    file_name_no_ext = os.path.splitext(filename)[0]

    # 计算文件名长度，注意中文字符
    filename_length = len(filename.encode('utf-8'))

    # 如果文件名长度超过最大长度限制
    if filename_length > max_length:
        # 生成一个时间戳标记
        timestamp = str(int(time.time()))
        # 截取文件名
        over_len = (filename_length-max_length)/3 * 3
        file_name_no_ext = file_name_no_ext[:-over_len]
        print(file_name_no_ext)
        new_filename = file_name_no_ext + "_" + timestamp + file_ext
    else:
        new_filename = filename

    return new_filename

truncate_filename("我是一个人.txt"*10,max_length=30)