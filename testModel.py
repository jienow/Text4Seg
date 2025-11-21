import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.eval.utils import decode_mask, translate_sequence, compute_logits_from_mask, masks_sample_points
from llava.model.segment_anything import SamPredictor, sam_model_registry

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

import os

# ===========================
# 配置
# ===========================
model_path = "./checkpoints/llava-v1.5-7b-p16"
image_path = "./images/horses.jpg"
expression = "Please segment the white horse in this image."
save_path = "./output_test/pred_mask.png"
sam_ckpt = "./llava/model/segment_anything/sam_vit_h_4b8939.pth"

# ===========================
# 加载 LLaVA + p16
# ===========================
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava"
)
model = model.eval().cuda()

# SAM
sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
sam = sam.cuda()
predictor = SamPredictor(sam)

# ===========================
# 构造 prompt
# ===========================
qs = f"{DEFAULT_IMAGE_TOKEN}\nSegment the object referred to as: {expression}"

conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

# ===========================
# 图像处理
# ===========================
image = Image.open(image_path).convert("RGB")
w_ori, h_ori = image.size

image_new = image.resize((336, 336))
image_tensor = process_images([image_new], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).cuda().to(model.dtype)

predictor.set_image(np.array(image))

# p16 → 16×16 grid
h, w = 16, 16

# ===========================
# 模型推理
# ===========================
output_ids = model.generate(
    input_ids,
    images=[image_tensor],
    image_sizes=[(336, 336)],
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.2,
)

text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

# ===========================
# 解析 <seg>...</seg>
# ===========================
try:
    mask_numbers = text_output.split("<seg>")[1].split("</seg>")[0]
    mask_numbers = decode_mask(mask_numbers)
    pred_mask = translate_sequence(mask_numbers)
except:
    pred_mask = [0] * (h * w)

if len(pred_mask) < h*w:
    pred_mask += [pred_mask[-1]] * (h*w - len(pred_mask))

pred_mask = torch.tensor(pred_mask).reshape(h, w)
pred_mask = (pred_mask > 0).float()

# 上采样
mask_upsample = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0),
    size=(h_ori, w_ori),
    mode='nearest').squeeze(0).squeeze(0)

# ===========================
# SAM 精修
# ===========================
if pred_mask.sum() == 0:
    sam_mask = np.zeros((h_ori, w_ori))
else:
    logits = compute_logits_from_mask(pred_mask.double())
    point_coords, point_labels = masks_sample_points(mask_upsample)

    sam_mask, _, logit = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=logits,
        multimask_output=False
    )

sam_mask = sam_mask[0].astype("uint8") * 255
os.makedirs(os.path.dirname(save_path), exist_ok=True)
Image.fromarray(sam_mask).save(save_path)

print("Saved:", save_path)
