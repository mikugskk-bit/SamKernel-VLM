from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import base64

import numpy as np

import cv2


import json

from tqdm import *



def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        # 解析JSON
        data = json.loads(text)
        boxs = []
        labels = []
        
        for item in data:
            if "bbox_2d" in item:
                x, y,xm,ym = item["bbox_2d"]
                boxs.append([x, y,xm,ym])
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(boxs)}")
                labels.append(label)
        
        return boxs, labels
        
    except Exception as e:
        print(f"Error: {e}")
        return [], []



qwen_model = ChatOpenAI(
    model="qwen2.5vl:7b",
    openai_api_key="000000", #"替换为自己的sk-秘钥"
    openai_api_base="http://172.31.234.152:11414/v1",
    temperature=0.5, # 降低温度以获得更确定的分析结果
    timeout=600
)

batch_data = np.load('newDataset/osem-50e4-30-1-train.npz')['osem']
boxes = []

qbar = trange(len(batch_data))
for i in range(len(batch_data)):
    raw_data = batch_data[i].reshape(128,128) * 256
    # data = cv2.imencode('.jpg', raw_data)[1]

    # 使用 JET 色彩映射
    pseudo_color_image = cv2.applyColorMap(raw_data.astype(np.uint8), cv2.COLORMAP_JET)
    data = cv2.imencode('.jpg', pseudo_color_image)[1]
    # cv2.imwrite('./pseudo_color_image.jpg', pseudo_color_image)

    image_bytes = data.tobytes()
    base64_encoded_data = base64.b64encode(image_bytes)
    image_data = base64_encoded_data.decode("utf-8")


    ex_data = cv2.imread('example_img.jpg')
    data = cv2.imencode('.jpg', ex_data)[1]
    image_bytes = data.tobytes()
    base64_encoded_data = base64.b64encode(image_bytes)
    examples_data = base64_encoded_data.decode("utf-8")

    user_message = HumanMessage(content=[
        {"type": "text", "text": "The following image is an example of brain bounding: a PET image of a human head, with the brain highlighted within a red box."},
        {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{examples_data}"},
        },
        {"type": "text", "text": "The brain should locate in the center of this PET image.Please locate the brain on following PET image with bounding box ([xmin , ymin , xmax , ymax]), Output the point coordinates in JSON format."},
        {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        }
    ])

    res = qwen_model.invoke([user_message]).content
    # print(res)
    box,label = decode_json_points(res)
    print(box)
    if box ==[]:
        box=[[-1,-1,-1,-1]]
    boxes.append(box[0])
    qbar.update(1)

np.savez('./newDataset/osem-50e4-30-1-box-train.npz',boxes=boxes)



batch_data = np.load('newDataset/osem-50e4-30-1-test.npz')['osem']
boxes = []
qbar = trange(len(batch_data))
for i in range(len(batch_data)):
    raw_data = batch_data[i].reshape(128,128) * 256
    # data = cv2.imencode('.jpg', raw_data)[1]

    # 使用 JET 色彩映射
    pseudo_color_image = cv2.applyColorMap(raw_data.astype(np.uint8), cv2.COLORMAP_JET)
    data = cv2.imencode('.jpg', pseudo_color_image)[1]
    # cv2.imwrite('./pseudo_color_image.jpg', pseudo_color_image)

    image_bytes = data.tobytes()
    base64_encoded_data = base64.b64encode(image_bytes)
    image_data = base64_encoded_data.decode("utf-8")


    ex_data = cv2.imread('example_img.jpg')
    data = cv2.imencode('.jpg', ex_data)[1]
    image_bytes = data.tobytes()
    base64_encoded_data = base64.b64encode(image_bytes)
    examples_data = base64_encoded_data.decode("utf-8")

    user_message = HumanMessage(content=[
        {"type": "text", "text": "The following image is an example of brain bounding: a PET image of a human head, with the brain highlighted within a red box."},
        {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{examples_data}"},
        },
        {"type": "text", "text": "The brain should locate in the center of this PET image.Please locate the brain on following PET image with bounding box ([xmin , ymin , xmax , ymax]), Output the point coordinates in JSON format."},
        {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        }
    ])

    res = qwen_model.invoke([user_message]).content
    # print(res)
    box,label = decode_json_points(res)
    print(box)
    if box ==[]:
        box=[[-1,-1,-1,-1]]
    boxes.append(box[0])
    qbar.update(1)

np.savez('./newDataset/osem-50e4-30-1-box-test.npz',boxes=boxes)

