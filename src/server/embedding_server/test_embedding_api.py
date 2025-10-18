import requests
import json


def test_rerank_service():
    """测试rerank服务是否正常工作"""
    url = "http://localhost:9001/embedding"

    # 测试数据
    data = {
        "texts": [
            "如何在Rust中实现内存分配？",
            "用Rust写一个加法函数的步骤是什么？",
            "What is the difference between stack and heap in Rust?",
            "Explain memory safety in Rust programming language."
        ]
    }

    try:
        print("发送请求到embedding服务...")
        print(f"URL: {url}")
        print(f"数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

        response = requests.post(url, json=data, timeout=30)

        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        print()

        if response.status_code == 200:
            result = response.json()
            print(f"embedding结果: {result}")
            return True
        else:
            print(f"请求失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    test_rerank_service()
