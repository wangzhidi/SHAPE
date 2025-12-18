import base64
from openai import OpenAI

client = OpenAI(
    api_key="sk-4g3yl3MOOGnTA8UrfaSpsImcVdFc42uXHVNbVefhykJEkT6N",
    base_url="https://cjp.bt6.top/v1"
)

# 假设你的图片文件名为 'image.png'
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 获取图片的Base64编码
image_base64 = encode_image("0_sit.png") 
text = "The current task is action editing. You need to modify the original action description text based on a reference sketch to generate new description text. Some examples are as follows:\n Example 1:\nAction Length: 120 frames\nOriginal Text: a person is clapping their hands\nReference Pose: [a sketch of a person running]\nSection to Edit: 40-80 frames\nResult: a person runs while clapping hands \nExample 2:\nAction Length: 120 frames\nOriginal Text: a man is standing still and then starts walking forward\nReference Pose: [a sketch of a person waving]\nSection to Edit: 40-80 frames\nResult: a person is standing still, then waves and walks forward \nNote: You do not need to return your thought process; just return the modified text description directly."
current_text="Current Task:\nAction Length: 120 frames\nOriginal Text: a person slowly leans back to the left and then lifts their right shoulder upwards.\nReference Pose: in the sketch picture\nSection to Edit: 60-100 frames\n \nReturn the modified description text only."


response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    n=1,
    messages=[
        {"role": "system", "content": text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": current_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message)

if response.usage:
    print("\n--- Token 消耗 ---")
    print(f"提示词 Tokens (prompt_tokens): {response.usage.prompt_tokens}")
    print(f"完成 Tokens (completion_tokens): {response.usage.completion_tokens}")
    print(f"总 Tokens (total_tokens): {response.usage.total_tokens}")
else:
    print("\n无法获取Token消耗信息。该第三方服务可能未提供此数据。")