import json

with open('/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_refer.json') as f:
    data = json.load(f)
    print(len(data))