import requests
import json

MODEL = "meta-llama/Meta-Llama-3.2-90B-Vision-Instruct"

def make_request(endpoint, data, BASE_URL):
    """
    Sends a POST request to the Llama model server.

    Args:
        endpoint (str): API endpoint to call.
        data (dict): JSON payload to send.

    Returns:
        dict: JSON response from the server, or None if the request fails.
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Content: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error in making request: {e}")
        return None

def get_llama_response(messages, url="http://localhost:9095"):
    """
    Gets a response from the Llama model for a chat-like interaction.

    Args:
        messages (list): List of message dictionaries in the format:
            [
                {"role": "system", "content": "System message here."},
                {"role": "user", "content": "User message here."},
                ...
            ]

    Returns:
        str: The model's response as a string.
    """
    data = {
        "model": MODEL,
        "messages": messages
    }
    response = make_request("/v1/chat/completions", data, url)
    if response and "choices" in response:
        return response["choices"][0]["message"]["content"]
    return "Error: No response from model."

if __name__ == "__main__":
    # Example usage
    example_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ]
    response = get_llama_response(example_messages)
    print("Response from Llama:", response)
