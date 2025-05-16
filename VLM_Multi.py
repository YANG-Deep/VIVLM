from openai import OpenAI
from PIL import Image
import io
import base64


def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None

input_image_path = "/Users/apple/Downloads/VLM/construction.png"
base64_image=convert_image_to_webp_base64(input_image_path)

client = OpenAI(api_key="sk-kcxtlkbqhnbbbmydpidfaufpgpshgdtdzedumygyyqgkvfdo", 
                base_url="https://api.siliconflow.cn/v1")

content = ""
reasoning_content=""

# 1. System Prompt:
# 你是一位聪明且逻辑严谨的交通评估者和调度者，拥有丰富的驾驶经验和场景理解能力，可以基于路侧摄像头的图片，识别到场景中的关键事件，并分析它们将会对车辆产生的影响，最后提供合理且逻辑清晰的指引车辆行驶的方法，以应对关键事件。
messages=[
    {"role": "system", "content": "你是一位聪明且逻辑严谨的交通评估者和调度者，拥有丰富的驾驶经验和场景理解能力，可以基于路侧摄像头的图片，识别到场景中的关键事件，并分析它们将会对车辆产生的影响，最后提供合理且逻辑清晰的指引车辆行驶的方法，以应对关键事件。"},
    {
    "role": "user",
    "content":[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail":"low"
            }
        },
        {
            "type": "text",
            "text": "请从所给图片中提取关键信息，对交通场景进行描述。\
                提取的关键信息可以分为环境条件和关键事件。环境条件信息包括天气、时间、道路环境和车道条件和车道数量以及方向等，\
                一个示例为：天气：晴朗、时间：白天、道路环境：城市道路，两旁是高大建筑物，有斑马线和交通标识，路面平整、\
                车道条件和数量：双向车道，车道划分清晰，双向四车道。关键事件描述整个驾驶环境决策相关的事件，比如潜在的碰撞风险和关键车辆(救护车，救火车等)，\
                通常以关键对象 + 动词/形容词 + 对交通环境的影响的格式给出。\
                关键事件的两个示例是：一个施工区域阻挡了右侧车道，因此汽车需要绕行至左侧第二车道以避免它/一个救火车位于左侧第二车道，\
                因此位于左侧第二车道的车辆需要切换至左侧第三车道或左侧第一车道，以保证救火车的快速通行。\
                现在请根据所给的交通图片和上面描述结构，输出对应的描述。"
        }
    ]}

]   
response = client.chat.completions.create(
    # model='Pro/deepseek-ai/DeepSeek-R1',
    model="deepseek-ai/deepseek-vl2",
    messages=messages,
    # temperature=0.7,  #平衡创造性与可靠性 
    # max_tokens=1000  # 单词请求最大生成长度  
    # stop=["\n##", "<|end|>"]  # 终止序列，在返回中遇到数组中对应的字符串，就会停止输出 
    # frequency_penalty=0.5  # 抑制重复用词（-2.0~2.0）  
    # stream=true # 控制输出是否是流式输出，对于一些输出内容比较多的模型，建议设置为流式，防止输出过长，导致输出超时
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
        # print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
        # print(chunk.choices[0].delta.reasoning_content, end="", flush=True)

# Round 2
messages.append({"role": "assistant", "content": content})
messages.append({'role': 'user', 'content': "请根据现场情况指挥交通"})
response = client.chat.completions.create(
    model="deepseek-ai/deepseek-vl2",
    messages=messages,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)