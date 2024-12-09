import requests
from openai import OpenAI

def generate_response(query, api_key="You'll need your own for now - sorry about that"):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    print(f'{"Type: " + str(type(completion.choices[0].message))}')
    print(completion.choices[0].message)
    return completion.choices[0].message

    # response = requests.post(url, headers=headers, json=data)
    # if response.status_code != 200:
    #     print("Status Code:", response.status_code)
    #     print("Response Text:", response.text)

    # response.raise_for_status()
    # return response.text