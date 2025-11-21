import os
import json
import random
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from question_answer_list import (
    QUESTION_ALL,
    QUESTION_PARTIAL,
    QUESTION_CONDITION,
    ANSWER_ALL,
    ANSWER_PARTIAL,
    ANSWER_CONDITION,
)
from utils import encode_mask, random_crop

h_new = w_new = 32
images_path = "./playground/data/coco/train2017"
annotations_path = "./playground/data/coco_stuff/annotations"

# 读取 cls_coco_stuff.txt 得到 labels
labels = {}
with open("./playground/data/create_json/cls_coco_stuff.txt") as f:
    for idx, line in enumerate(f):
        labels[idx] = line.strip()


def process_single(args):
    """
    单个任务的处理逻辑（保证和你原来的 for 循环里逻辑一致）。
    args: (rep_idx, image_file)，rep_idx 只是占位，保持和原来外层 for _ in range(10) 语义一致
    """
    rep_idx, image_file = args  # rep_idx 当前不直接使用，只是保持逻辑结构

    item = {}
    item["id"] = image_file.split(".")[0]
    item["image"] = "coco/train2017/" + image_file

    # 读取 annotation
    annotation_path = os.path.join(
        annotations_path, image_file.replace(".jpg", "_labelTrainIds.png")
    )

    # 有可能有缺失标注文件，这里做个保护
    if not os.path.exists(annotation_path):
        return None

    annotation = Image.open(annotation_path)

    # 随机裁剪（你的 random_crop）
    annotation, crop_params = random_crop(annotation)

    item["crop_size"] = crop_params
    item["height"] = h_new
    item["width"] = w_new

    # resize 到 32x32
    annotation = annotation.resize((w_new, h_new), Image.NEAREST)
    array_annotation = np.array(annotation)
    array_annotation = array_annotation.flatten()

    # label -> 类别名，255 视为 others
    label_each_pixel = [
        labels[label].split(", ")[0] if label in labels else "others"
        for label in array_annotation
    ]

    labels_in_image = np.unique(array_annotation)

    # 如果整张图只有一个类别，直接跳过（和原逻辑一致）
    if len(labels_in_image) == 1:
        return None

    # 去掉 255
    labels_in_image = labels_in_image[labels_in_image != 255]
    labels_not_in_image = [label for label in range(171) if label not in labels_in_image]

    # 对话构造
    conversation_list = []
    question_all = False

    for round_idx in range(2):
        # 选择问题类型
        random_number = random.random()

        if question_all:
            random_number = random.choice([0.4, 0.9])

        if random_number < 0.1:
            # QUESTION_ALL 分支
            question_all = True
            question = random.choice(QUESTION_ALL)
            if round_idx == 0:
                conversation_list.append(
                    {"from": "human", "value": f"<image>\n{question}"}
                )
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            # 构造多样化 label 映射
            label_diversity = labels.copy()
            for label in labels_in_image:
                if random.random() < 0.2:
                    label_diversity[label] = labels[label].split(", ")[0]
                else:
                    try:
                        label_diversity[label] = random.choice(
                            labels[label].split(", ")[1:]
                        )
                    except Exception:
                        label_diversity[label] = labels[label].split(", ")[0]

            # 255 -> others
            label_each_pixel_diversity = [
                label_diversity[label] if label in label_diversity else "others"
                for label in array_annotation
            ]

            # ---- index ----
            label_each_pixel_diversity = np.reshape(
                label_each_pixel_diversity, (h_new, w_new)
            )
            SEG = encode_mask(label_each_pixel_diversity)
            # ---- index ----

            answer = random.choice(ANSWER_ALL) + f"\n<seg>{SEG}</seg>"
            conversation_list.append({"from": "gpt", "value": answer})

        elif random_number < 0.4:
            # QUESTION_PARTIAL 分支
            question = random.choice(QUESTION_PARTIAL)

            partial_labels_ids = random.sample(
                labels_in_image.tolist(), random.randint(1, len(labels_in_image))
            )

            partial_labels = [
                labels[partial_label].split(", ")[0]
                for partial_label in partial_labels_ids
            ]

            random.shuffle(partial_labels)
            question = question.replace("[class_name]", "|".join(partial_labels))

            if round_idx == 0:
                conversation_list.append(
                    {"from": "human", "value": f"<image>\n{question}"}
                )
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            label_each_pixel_partial = [
                label if label in partial_labels else "others"
                for label in label_each_pixel
            ]

            # ---- index ----
            label_each_pixel_partial = np.reshape(
                label_each_pixel_partial, (h_new, w_new)
            )
            SEG = encode_mask(label_each_pixel_partial)
            # ---- index ----

            answer = (
                random.choice(ANSWER_PARTIAL).replace(
                    "[class_name]", "|".join(partial_labels)
                )
                + f"\n<seg>{SEG}</seg>"
            )
            conversation_list.append({"from": "gpt", "value": answer})

        else:
            # QUESTION_CONDITION 分支
            question = random.choice(QUESTION_CONDITION)

            condition_labels_ids = random.sample(
                labels_in_image.tolist(), random.randint(1, len(labels_in_image))
            )

            condition_labels = [
                labels[condition_label].split(", ")[0]
                for condition_label in condition_labels_ids
            ]

            # 随机加入 "others"
            if random.random() < 0.5:
                condition_labels.append("others")

            # 加一些冗余类别（不在图片中的类别）
            if random.random() < 0.9 and len(labels_not_in_image) > 0:
                redundant_labels_ids = random.sample(
                    labels_not_in_image, random.randint(1, len(labels_not_in_image))
                )
                redundant_labels = [
                    labels[redundant_label].split(", ")[0]
                    for redundant_label in redundant_labels_ids
                ]
                condition_redundant_labels = condition_labels + redundant_labels
            else:
                condition_redundant_labels = condition_labels

            random.shuffle(condition_redundant_labels)
            question = question.replace(
                "[class_name]", "|".join(condition_redundant_labels)
            )

            if round_idx == 0:
                conversation_list.append(
                    {"from": "human", "value": f"<image>\n{question}"}
                )
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            label_each_pixel_condition = [
                label if label in condition_labels else "others"
                for label in label_each_pixel
            ]

            # ---- index ----
            label_each_pixel_condition = np.reshape(
                label_each_pixel_condition, (h_new, w_new)
            )
            SEG = encode_mask(label_each_pixel_condition)
            # ---- index ----

            answer = (
                random.choice(ANSWER_CONDITION).replace(
                    "[class_name]", "|".join(condition_redundant_labels)
                )
                + f"\n<seg>{SEG}</seg>"
            )
            conversation_list.append({"from": "gpt", "value": answer})

        # 和原代码一样，每轮都覆盖一次，最后得到两轮的列表
        item["conversations"] = conversation_list

    return item


def main():
    # 保持和原来一样，先拿到所有 image_file
    image_files = os.listdir(images_path)

    # 原来是: for _ in range(10): for image_file in ...
    # 现在展开成任务列表，保证顺序是：
    # rep 0: 所有图片
    # rep 1: 所有图片
    # ...
    tasks = [
        (rep_idx, image_file)
        for rep_idx in range(10)
        for image_file in image_files
    ]

    Content = []

    # 多进程 + 全局进度条
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(
            pool.imap(process_single, tasks), total=len(tasks), desc="Processing"
        ):
            if result is not None:
                Content.append(result)

    # 写出 json
    os.makedirs("./playground/data/json_files", exist_ok=True)
    with open("./playground/data/json_files/cocostuff_32_two_round_10.json", "w") as f:
        json.dump(Content, f, indent=4)


if __name__ == "__main__":
    main()
