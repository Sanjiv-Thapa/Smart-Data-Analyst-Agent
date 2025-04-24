import pandas as pd
from openai import OpenAI
from prompts import system_prompt
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def basic_eda(df):
    desc = df.describe(include='all').to_string()
    info = f"shape{df.shape}\n\nmissingvalues:\n{df.isnull().sum()}\n\nData types:\n{df.dtypes}"
    return f"{info}\n\nsummary:\n{desc}"

def ask_llm(eda_summary):
    system_prompt = "You are a helpful data science assistant"  # Define your system prompt
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":eda_summary}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content  # Updated response access

def answer_question(df, question):
    sample = df.head(3).to_csv(index=False)
    prompt = f"Here is a sample of the data:\n{sample}\n\nQuestion:{question}"
    return ask_llm(prompt)
    # messages = [
    #     {'role':"system","content":"You are a helpful data science assistant"},
    #     {'role':"user","content":prompt}
    # ]
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     temperature=0.7
    # )
    # return response.choices[0].message.content
def suggest_next_action():
    prompt = "Given EDA has been done and preprocessing suggestions provided, what should I do next to analyze this dataset?"
    return ask_llm(prompt)

def run_generated_code(df, prompt):
    full_prompt = f"Given the following data sample, write Python code to {prompt}.\n\nData sample:\n{df.head(3).to_csv(index=False)}"
    code = ask_llm(full_prompt)
    try:
        exec_globals = {'pd': pd, 'df': df.copy()}
        exec(code, exec_globals)
        return exec_globals.get('df', df), code
    except Exception as e:
        return df, f"Error running generated code: {e}\n\nCode:\n{code}"
