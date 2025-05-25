import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

response = client.invoke_model(
    modelId="anthropic.claude-opus-4-20250514-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "こんにちは。自己紹介してください。"
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }),
    contentType="application/json"
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"])
