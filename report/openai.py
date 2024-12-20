import os
import json
import requests

openai_api_key = os.environ.get('OPENAI_API_KEY')
url = 'https://api.openai.com/v1/chat/completions'
model_name = 'gpt-4'

system_prompt = (
    "You are a code documentation assistant. "
    "The user will provide MATLAB code, and your task is to produce a well-documented version of the code. "
    "Add descriptive comments, section headings (%%) where appropriate, explain the purpose of functions and scripts, and ensure that the code is ready for MATLAB's publish or report generator. "
    "Do not remove or alter code functionality, just improve documentation and structure. "
    "Return the fully annotated code."
)

parent_dir = os.path.join('..')
m_files = [f for f in os.listdir(parent_dir) if f.endswith('.m')]

if not m_files:
    print('No .m files found in the parent directory.')
else:
    for filename in m_files:
        filepath = os.path.join(parent_dir, filename)
        print(f'Processing {filename}...')

        with open(filepath, 'r') as f:
            code_text = f.read()

        request_body = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': code_text}
            ],
            'max_tokens': 4000,
            'temperature': 0.3,
            'n': 1
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }

        try:
            response = requests.post(url, headers=headers, json=request_body, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f'OpenAI API request failed for {filename}: {e}')
            continue

        data = response.json()
        if 'choices' in data and data['choices']:
            improved_code = data['choices'][0]['message']['content']
        else:
            print(f'No valid response returned from OpenAI for file {filename}.')
            continue

        new_filename = f'report_{filename}'
        with open(new_filename, 'w') as f:
            f.write(improved_code)

        print(f'Improved code written to {new_filename}')

    print('Processing complete.')
