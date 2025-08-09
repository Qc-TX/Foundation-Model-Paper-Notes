---
description: 2025 DataWhale夏令营 第三期
---

# 夏令营：AI图像生成与攻防技术

**一、题目要求与核心任务**

根据赛题要求，我们主要需完成四大子任务：

1. **AIGC图片生成**：基于文本Prompt生成逼真图像
2. **自然场景编辑**：实现指定区域的图像内容修改
3. **视觉文本编辑**：修改图像中的文字信息
4. **Deepfake**：实现跨人脸图像交换

**整体赛题任务有3个难点：**

* 生成内容需通过VIEScore（SC+PQ）双重验证,可以理解为sc是估生成图像与输入文本提示的语义对齐程度，以及图像内部元素的逻辑自洽性；pq是指生成图像的视觉保真度和真实性，对抗现有防伪系统的鲁棒性。
  * ```
    sc_score = clip_similarity(prompt_embedding, image_embedding) * spatial_coherence_score
    ```
  * ```
    pq_score = frechet_inception_distance(real_faces, generated_faces) * anti_detection_score
    ```
* Deepfake需对抗现有防伪系统识别
* 如何采用多进程高并发的去实现整个代码设计，优化效率同时提高生成质量(因为只有两次提交机会/天，多模型的时候本地筛选一下)

**二、关键技术改进与创新尝试**

**Task2与baseline中的代码比较：**

| 维度         | 基线代码（Baseline）             | 改进代码（Improved）                   |
| ---------- | -------------------------- | -------------------------------- |
| **模型架构**   | 使用CogView4Pipeline（本地部署模型） | 集成阿里云DashScope API（云端高性能模型）      |
| **任务处理策略** | 简单分类处理（t2i/deepfake）       | 分任务精细化处理（t2i/tie/vttie/deepfake） |
| **抗失败机制**  | 无重试机制                      | 下载图片增加3次重试                       |
| **代码结构**   | 功能集中式实现                    | 模块化设计（下载/生成/人脸交换分离）              |
| **错误处理**   | 无异常捕获                      | 关键步骤添加try-except日志记录             |

**1、baseline分数低的原因：**&#x672C;地下载的模型，生成质量有限，且仅仅只生成了10个，因此我们提交之后只有0.4左右的分数。

**2、模型升级：**&#x672C;地部署的CogView4模型切换至阿里云模型API：

```
#使用通义万相Text2Image API
def convert_text_prompt_to_image(prompt):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    data = {"model": "wanx2.1-t2i-turbo", "input": {"prompt": prompt}}
```

**3、任务专业化处理：**

**视觉文本编辑（vttie）**:

```
#调用描述编辑专用接口
def image_edit_by_prompt(function_str, prompt, base_image_url):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
    data = {
        "model": "wanx2.1-imageedit",
        "input": {"function": function_str, "prompt": prompt, "base_image_url": base_image_url}
    }
    ...
```

**4、性能优化**：

* **异步任务管理**：通过任务ID轮询状态，避免阻塞主线程
* **分布式下载**：使用多线程加速图片素材获取

**\*\*三、TASK2的代码学习：**

1.**容错机制**：

```
# 图片下载重试机制
def download_image_with_retry(url, max_retry=3):
    for attempt in range(max_retry):
        try:
            return download_image(url)
        except Exception as e:
            if attempt < max_retry-1:
                time.sleep(2**attempt)  # 指数退避策略
            else:
                raise
```

2、**分层异常捕获**

```
def download_image(url):
    try:
        response = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
```

**四、继续改进优化：**

**1、按照TASK3的改进，我增加了提示词优化和多模型两个部分。**

具体的代码：

```
# 修改主循环部分
for idx, row in task.iterrows():
    if os.path.exists("imgs/" + str(row['index']) + ".jpg"):
        continue
​
    url = "http://mirror.coggle.club/tianchi/532389/imgs/01309910e64f48caaffe5b3db8b344e3.jpg"
    download_image(url, "./imgs/", str(row['index']) + ".jpg")
​
    # 尝试次数，用于模型选择
    max_attempts = 3
    success = False
    
    for attempt in range(1, max_attempts + 1):
        try:
            if row.task_type == "t2i":
                # 优化提示词
                optimized_prompt = optimize_prompt(row.prompt)
                # 选择模型
                model = select_model("t2i", attempt)
                print(f"使用模型 {model} 处理任务 {row['index']}")
                url = convert_text_prompt_to_image(optimized_prompt, model=model)
            
            elif row.task_type == "tie":
                optimized_prompt = optimize_prompt(row.prompt)
                model = select_model("tie", attempt)
                print(f"使用模型 {model} 处理任务 {row['index']}")
                url = image_edit_by_prompt("stylization_all", optimized_prompt, 
                                          "http://mirror.coggle.club/tianchi/532389/imgs/" + row.ori_image, 
                                          model=model)
            
            elif row.task_type == "vttie":
                optimized_prompt = optimize_prompt(row.prompt)
                model = select_model("vttie", attempt)
                print(f"使用模型 {model} 处理任务 {row['index']}")
                url = image_edit_by_prompt("description_edit", optimized_prompt, 
                                          "http://mirror.coggle.club/tianchi/532389/imgs/" + row.ori_image, 
                                          model=model)
            
            if url:
                download_image(url, "./imgs/", str(row['index']) + ".jpg")
                success = True
                break
        except Exception as e:
            print(f"尝试 {attempt} 失败: {e}")
            continue
    
    if not success:
        url = "http://mirror.coggle.club/tianchi/532389/imgs/01309910e64f48caaffe5b3db8b344e3.jpg"
        download_image(url, "./imgs/", str(row['index']) + ".jpg")
​
    # Deepfake任务保持不变
    if row.task_type == "deepfake":
        try:
            face_swap_using_dlib(
                "./data/imgs/" + row['ori_image'], 
                "./data/imgs/" + row['target_image'],
                "./imgs/" + str(row['index']) + ".jpg"
            )
        except:
            url = "http://mirror.coggle.club/tianchi/532389/imgs/01309910e64f48caaffe5b3db8b344e3.jpg"
            download_image(url, "./imgs/", str(row['index']) + ".jpg")
```

2、替换Qwen-Image

```
from diffusers import DiffusionPipeline
import torch
​
model_name = "Qwen/Qwen-Image"
​
# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"
​
pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)
​
positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition." # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
}
​
# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
​
negative_prompt = " " # using an empty string if you do not have specific concept to remove
​
​
# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
​
width, height = aspect_ratios["16:9"]
​
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
​
image.save("example.png")
​
```



目前的分1.63,还在改进。\
