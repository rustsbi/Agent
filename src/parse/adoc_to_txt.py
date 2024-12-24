import subprocess
from bs4 import BeautifulSoup
import os
# 所解析的adoc文件中，若含有include，那么其include的文件也必须存在
def parse_adoc_to_txt(input_file, output_file):
    try:
        # 使用 Asciidoctor 将 AsciiDoc 转换为 HTML:
        subprocess.run(['asciidoctor', '-o', 'temp.html', input_file], check=True)
        
        # 读取生成的 HTML 文件
        with open('temp.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 使用 BeautifulSoup 提取纯文本
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # 写入纯文本文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 删除临时 HTML 文件
        os.remove('temp.html')
        
        print(f"转换成功！纯文本文件已保存为 {output_file}")
    except subprocess.CalledProcessError as e:
        print("Asciidoctor 转换失败。")
        print(e)
    except Exception as e:
        print("转换过程中出错。")
        print(e)

def get_adoc_files(directory):
    # 列出指定目录下的所有条目
    entries = os.listdir(directory)
    # 筛选出以 .adoc 结尾的文件
    adoc_files = [file for file in entries if file.endswith('.adoc') and os.path.isfile(os.path.join(directory, file))]
    return adoc_files

# 在https://github.com/riscv/riscv-isa-manual/tree/main/src目录下进行的测试
# 使用时所include的文档都可以找到
def test_adoc_to_txt():
    input1 = "riscv-privileged.adoc"
    input2 = "riscv-unprivileged.adoc"
    parse_adoc_to_txt(input1, input1.replace('.adoc', '.txt'))
    parse_adoc_to_txt(input2, input2.replace('.adoc', '.txt'))