import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

def get_claude_response(messages):
    """
    Interacts with the Claude 3.5 Sonnet model using the provided messages.

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
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    prompt = ""
    for message in messages:
        if message['role'] == 'system':
            prompt += f"{HUMAN_PROMPT} {message['content']}"
        elif message['role'] == 'user':
            prompt += f"{HUMAN_PROMPT} {message['content']}"
        elif message['role'] == 'assistant':
            prompt += f"{AI_PROMPT} {message['content']}"

    response = client.completions.create(
        model="claude-3.5-sonnet",
        prompt=prompt,
        max_tokens_to_sample=300
    )

    return response.completion.strip()

if __name__ == "__main__":
    # Example usage
    example_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ]
    response = get_claude_response(example_messages)
    print(response)
