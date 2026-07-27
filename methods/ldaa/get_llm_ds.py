

API_KEY = "" 

base_url = ""

import requests
import time

def get_query_1(query, max_retries=3):
    
    retries = 0
    global error_count
    messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ]
    while retries < max_retries:
        print(f"process {messages}: {messages}")
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "deepseek-v3-250324", 
                    "messages": messages
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=10  # 添加超时设置
            )
            
            if response.status_code == 200:
                completion = response.json()
                new_response = completion["choices"][0]["message"]["content"]
                return new_response
            else:
                retries += 1
                print(f"fail, status {response.status_code}, try ({retries}/{max_retries})...")
                time.sleep(10)  # 等待30秒后重试
                
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Exception: {e}, retry ({retries}/{max_retries})...")
            time.sleep(10)
            return "Exception"
    
    # 如果重试后仍然失败
    print(f"fail after {max_retries} try. Set 'None' for {messages}")
    return "None"