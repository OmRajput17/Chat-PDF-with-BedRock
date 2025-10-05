import boto3
import json

from botocore.exceptions import ClientError

prompt_data = """
Act as a Shakespeare and write a poem on Generative AI.
"""

bedrock = boto3.client(service_name = "bedrock-runtime")

# Embed the prompt in Llama 3's instruction format.
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt_data}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9,
}

## Convert the native request to Json
request = json.dumps(native_request)

model_id = "meta.llama3-70b-instruct-v1:0"

try:
    # Invoke the model with the request.
    response = bedrock.invoke_model(
        modelId = model_id, 
        body = request,
        accept = "application/json",
        contentType = "application/json",
    )

except(ClientError, Exception) as e:
    print(f"ERROR: Can't invoke'{model_id}'. Reason: {e}")
    exit(1)

### Decode the response body
model_response = json.loads(response["body"].read())


response_text = model_response["generation"]
print(response_text)