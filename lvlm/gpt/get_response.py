import os
from openai import OpenAI

def get_gpt_response(messages):
    """
    Interacts with the GPT-4o-mini model using the provided messages.

    Args:
        messages (list): A list of message dictionaries in the format:
            [
                {"role": "system", "content": "System message here."},
                {"role": "user", "content": "User message here."},
                ...
            ]

    Returns:
        str: The assistant's response as a string.
    """

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    example_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ]
    response = get_gpt_response(example_messages)
    print(response)
