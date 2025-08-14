import requests
import json


def test_rerank_service():
    """测试rerank服务是否正常工作"""
    url = "http://localhost:8001/rerank"

    # 测试数据
    data = {
        "query": "什么是机器学习？",
        "passages": [
            "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的学习过程。",
            "自然语言处理是人工智能的一个重要应用领域，专注于计算机理解和生成人类语言。"
        ]
    }

    try:
        print("发送请求到rerank服务...")
        print(f"URL: {url}")
        print(f"数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

        response = requests.post(url, json=data, timeout=30)

        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            result = response.json()
            print(f"Rerank结果: {result}")
            return True
        else:
            print(f"请求失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    test_rerank_service()
