"This script uses an OpenAI API to label a random sample of 100,000 tweets from the dataset in 19 topics"

import sys
import json
import openai
import os
import pandas as pd
import numpy as np
from pprint import pprint
import datetime
import time
import tiktoken
import ast

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "gpt_label_tweets.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "gpt_label_tweets_error.log"), "a")

# Redirect stdout and stderr
sys.stdout = log_file
sys.stderr = error_log_file

print("This will be logged in gpt_label_tweets.log")
sys.stdout.flush()

print("packages imported")
sys.stdout.flush()

# Set API info
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Set OPENAI_API_KEY before running the GPT labelling script.")
fine_tuned_model_id = "gpt-4o-mini"


def read_csv_from_local(filename):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(filename, engine="python")


if __name__ == "__main__":
    filename = os.path.join(RAW_DIR, "df_full_july_tweets.csv")
    data = read_csv_from_local(filename)
    print(data.head())
    sys.stdout.flush()

df = data.sample(n=100000, random_state=42)

system_message = """
You are a helpful research assistant.
I will give you a tweet from a politician around the world (which has been translated into English) and you must classify it into specific topics or return 'no topic' if none apply.

Here are the topics to classify into, along with brief descriptions. Use only the following labels, as each tweet must match exactly one or more of these labels:

1. **immigration**: Topics related to the movement of people across borders, including refugee issues and their integration into host countries.
2. **climate change**: Topics related to the long-term changes in global weather patterns and their environmental impact.
3. **renewable energy**: Topics related to energy sources that are sustainable and environmentally friendly, such as solar, wind power or electric vehicles.
4. **traditional energy**: Topics related to non-renewable energy sources like coal, oil, and natural gas.
5. **inequality**: Topics related to redistribution and disparities in wealth, income, or socio-economic status.
6. **social policy**: Topics related to policies providing social assistance to poor people and social insurance against income risk including pensions.
7. **taxation**: Topics involving taxes, government revenues, and related economic policies.
8. **labour market**: Topics related to employment, unemployment, job markets, and labor rights.
9. **international trade**: Topics related to the exchange of goods and services across borders.
10. **economics**: Topics related to the economy, fiscal policy, inflation, monetary policy, financial markets and business issues.
11. **european union**: Topics specifically related to the European Union, including its institutions and policies.
12. **public health**: Topics related to public health policy, health crises, and healthcare systems.
13. **gender rights**: Topics related to gender equality, women’s rights, abortion and LGBTQ+ issues.
14. **civil rights**: Topics related to individual freedoms and rights, including free speech and personal liberties.
15. **political rights**: Topics related to political freedoms such as voting, assembly, and political participation.
16. **connecting**: Tweets simply about personal communication such as greetings, weather or holidays.
17. **elections/voting**: Topics related to elections, voting or political campaigns.
18. **self-promotion**: When politicians promote themselves or their work e.g. "I'm on air today", "Check out my interview", "We have accomplished a lot during our term" etc.
19. **anti-establishment**: Covers all tweets which are
        (a) anti-system: idea that the current political system is broken, rigged and undemocratic.
        (b) anti–political elites: idea that political elites in elected office are irresponsible, dishonest, guilty of cronyism and only look after their own interests.
        (c) anti-technocratic: idea that political elites in non-elected institutions (e.g. civil servants, EU, IMF, WTO) are irresponsible, dishonest, guilty of cronyism and only look after their own interests.
        (d) generic anti-elite: idea that non-political elites (e.g. big business, millionaires, establishment media) are irresponsible, dishonest, guilty of cronyism and only look after their own interests.

Rules:
- Output only the labels.
- If a tweet fits multiple topics, list all the relevant topics.
- If the tweet does not fit any of these topics, output only "No topic".
- Be concise and accurate in matching the topics.
"""

tokens_used = 0
prompt_tokens = 0
completion_tokens = 0

initial_time = time.time()
start_time = time.time()

# Function ensures the ChatGPT outcome only ever takes one of the 19 values or "no topic"
functions = [
    {
        "name": "classify_tweet",
        "description": "Classifies the tweet based on specified topics or returns 'No topic' if none apply.",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "immigration",
                            "climate change",
                            "renewable energy",
                            "traditional energy",
                            "inequality",
                            "social policy",
                            "taxation",
                            "labour market",
                            "international trade",
                            "economics",
                            "european union",
                            "public health",
                            "gender rights",
                            "civil rights",
                            "political rights",
                            "connecting",
                            "elections/voting",
                            "self-promotion",
                            "anti-establishment",
                            "no topic",
                        ],
                    },
                    "description": "List of relevant topics or 'no topic' if not applicable.",
                }
            },
            "required": ["topics"],
        },
    }
]

print(len(df))
sys.stdout.flush()

counter = 0

for index, row in df.iterrows():
    counter += 1

    if counter % 100 == 0:
        print(f"Processing tweet {counter}")
        sys.stdout.flush()

    time.sleep(0.1)

    test_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["text_translate"]},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id,
            messages=test_messages,
            temperature=0,
            max_tokens=500,
            functions=functions,
            function_call={"name": "classify_tweet"},
        )

        if "function_call" in response["choices"][0]["message"]:
            function_call_arguments = response["choices"][0]["message"]["function_call"]["arguments"]
            topics = json.loads(function_call_arguments)["topics"]
        else:
            topics = ["No topic"]

        df.at[index, "gpt_finetuned"] = json.dumps(topics)

        tokens_used = tokens_used + response["usage"]["total_tokens"]
        prompt_tokens = prompt_tokens + response["usage"]["prompt_tokens"]
        completion_tokens = completion_tokens + response["usage"]["completion_tokens"]

    except json.JSONDecodeError:
        print(f"JSONDecodeError at row {index}, dropping this row.")
        df.drop(index, inplace=True)
        continue

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df.drop(index, inplace=True)
        continue

    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time > 55:
        print("Checking token rates")
        if tokens_used > 88000:
            print("Stopping for one minute to avoid rate limit")
            time.sleep(60)
            tokens_used = 0
        else:
            print("Continuing processing")
        start_time = time.time()

final_time = time.time()
total_time = final_time - initial_time
print(f"Execution time: {total_time} seconds")

print(f"Total tokens used: {tokens_used}")
print(f"Prompt tokens used: {prompt_tokens}")
print(f"Completion tokens used: {completion_tokens}")


def parse_topics(topics_str):
    try:
        topics = ast.literal_eval(topics_str)
        if isinstance(topics, list):
            return topics
    except (ValueError, SyntaxError):
        pass
    return []


df["gpt_finetuned"] = df["gpt_finetuned"].apply(parse_topics)


def clean_topics(topics):
    if "no topic" in topics and len(topics) > 1:
        topics = [topic for topic in topics if topic != "no topic"]
    return topics


df["gpt_finetuned"] = df["gpt_finetuned"].apply(clean_topics)

output_path = os.path.join(PROCESSED_DIR, "labelled_tweets_100000.csv")
df.to_csv(output_path, index=False)

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Saved GPT-labelled tweets to {output_path}")
