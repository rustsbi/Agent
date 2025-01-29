## 12月工作记录

12月工作主要集中在编写文档切片存向量数据库代码以及rst文档的解析代码。

### rst文档的解析代码

对于rst文档的解析切片，我尝试了两种方案。第一种是受zzh同学解析adoc文档代码的启发，使用docutils库将rst文件转成html文件，然后使用beautifulsoup将html文件解析成文本。源代码如下：

```python
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
document = publish_parts(content, writer_name='html')

soup = BeautifulSoup(document['html_body'], 'html.parser')
text = soup.get_text()
return text
```

第二种方案是使用re正则表达式解析rst文档，将rst文档中的标题和内容分开，同时进行分级，然后存入字典中，最后转成字符串形式。源代码如下：

```python
def parse_rst(file_path):
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        current_lines = []
        current_title = ""

        for content_line in content.split('\n'):
            # 去除回车行
            if content_line.strip() == '':
                continue
            title_match = re.search(r'^={3,}$', content_line)
            subtitle_match = re.search(r'^-{3,}$', content_line)
            if title_match or subtitle_match:
                if title_match:
                    level = 1
                else:
                    level = 2
                if len(current_lines) > 0:
                    new_title = current_lines[-1]
                    current_lines.pop()
                    if len(current_lines) > 0:
                        new_string = "\n".join(current_lines)
                        chunks.append({'title': current_title,'level': level,'content': new_string})
                    current_title = new_title
                    current_lines = []
            else:
                current_lines.append(content_line)

        if len(current_lines) > 0:
            new_string = "\n".join(current_lines)
            chunks.append({'title': current_title,'level': 1,'content': new_string})

        text_data = [f"{item['title']} (Level {item['level']}): {item['content']}" for item in chunks if 'title' in item and 'content' in item and 'level' in item]
        return text_data


    except Exception as e:
        print(f"Error reading .rst file: {e}")
        return ""
```

### 文档切片存向量数据库代码

我创建了files2db.py，主体内容来自原来的demo程序，程序功能为将files文件夹中所有的文档进行切片解析，存入db文件夹中。files2db.py的代码较长，这里不再展示。

### 数据库清除代码

我创建了clear_db_dirs.py，程序功能为清除db文件夹下的所有文档数据库文件夹，便于管理数据库。程序源码如下：

```python
import os
import shutil

# 指定要清理的目录
db_directory = 'db'

# 检查db_directory是否存在
if not os.path.exists(db_directory):
    print(f"The directory {db_directory} does not exist.")
else:
    # 遍历db_directory中的所有子目录
    for folder in os.listdir(db_directory):
        folder_path = os.path.join(db_directory, folder)
        
        # 确保是目录而不是文件
        if os.path.isdir(folder_path):
            # 检查index.faiss和index.pkl是否存在于子目录中
            if os.path.isfile(os.path.join(folder_path, 'index.faiss')) and os.path.isfile(os.path.join(folder_path, 'index.pkl')):
                # 删除子目录及其内容
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            else:
                print(f"Folder {folder_path} does not contain both index.faiss and index.pkl.")
```

### 未来工作

- rst文档解析函数还有进一步优化的空间，主要在对于文档的解析可以更加细粒度。

- demo程序可以更改逻辑，不需要强制用户去上传文件，而是调用files2db程序读取数据库的文档。