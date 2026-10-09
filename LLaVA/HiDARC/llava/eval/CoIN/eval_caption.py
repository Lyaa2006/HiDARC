import os
import argparse
import json
import re
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VizWiz/val_new.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim_new/VizWiz/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--max-caption-words', type=int, default=128)
    parser.add_argument('--max-caption-chars', type=int, default=512)
    return parser.parse_args()

def sanitize_caption(text, max_words=128, max_chars=512):
    text = re.sub(r"\s+", " ", str(text)).strip()
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        clipped = text[:max_chars]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        text = clipped.strip()
        truncated = True
    words = text.split()
    if max_words > 0 and len(words) > max_words:
        text = " ".join(words[:max_words])
        truncated = True
    return text, truncated


def create_coco_type(annotation_file, result_file, output_dir, max_words=128, max_chars=512):
    with open(result_file) as f:
        results = [json.loads(line) for line in f if line.strip()]

    total = len(results)
    if total == 0:
        raise ValueError(
            f"No predictions found in result file: {result_file}. "
            "The generation stage likely failed before caption evaluation."
        )
    coco_results = []
    image_id = 1
    truncated_predictions = 0
    for result in results:
        pred, was_truncated = sanitize_caption(
            result['text'], max_words=max_words, max_chars=max_chars
        )
        truncated_predictions += int(was_truncated)

        coco_results.append({
            "image_id": int(image_id),  # 确保 image_id 是整数类型
            "caption": pred
        })
        image_id += 1
    output_file = 'pred_coco_type.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total, truncated_predictions

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_coco_caption_annotation(
    annotation_file,
    result_file=None,
    output_dir=None,
    max_words=128,
    max_chars=512,
):
    data = load_json(annotation_file)

    result_image_ids = None
    if result_file is not None and os.path.exists(result_file):
        result_data = load_json(result_file)
        result_image_ids = {int(item["image_id"]) for item in result_data if "image_id" in item}

    annotations = data.get("annotations", [])
    needs_category_id = any("category_id" not in ann for ann in annotations)
    needs_categories = not data.get("categories")
    needs_subset = result_image_ids is not None and len(result_image_ids) > 0
    truncated_annotations = 0

    if not needs_category_id and not needs_categories and not needs_subset:
        for ann in annotations:
            _, was_truncated = sanitize_caption(
                ann.get("caption", ""),
                max_words=max_words,
                max_chars=max_chars,
            )
            truncated_annotations += int(was_truncated)
        if truncated_annotations == 0:
            return annotation_file, truncated_annotations

    normalized = dict(data)
    normalized_annotations = []
    for ann in annotations:
        if result_image_ids is not None and int(ann["image_id"]) not in result_image_ids:
            continue
        item = dict(ann)
        item.setdefault("category_id", 0)
        item["caption"], was_truncated = sanitize_caption(
            item.get("caption", ""),
            max_words=max_words,
            max_chars=max_chars,
        )
        truncated_annotations += int(was_truncated)
        normalized_annotations.append(item)
    normalized["annotations"] = normalized_annotations

    if "images" in normalized:
        normalized["images"] = [
            dict(image)
            for image in normalized["images"]
            if result_image_ids is None or int(image["id"]) in result_image_ids
        ]

    if needs_categories:
        normalized["categories"] = [{"id": 0, "name": "caption"}]

    if output_dir is None:
        output_dir = os.path.dirname(annotation_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    normalized_path = os.path.join(output_dir, "normalized_coco_caption_annotation.json")
    with open(normalized_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    return normalized_path, truncated_annotations

def merge_captions(pred_file, val_file, output_file):
    # 加载预测文件和验证文件
    pred_data = load_json(pred_file)
    val_data = load_json(val_file)

    # 将验证数据中的 captions 根据 image_id 组织成字典，image_id -> [captions]
    val_dict = defaultdict(list)
    for item in val_data['annotations']:
        val_dict[item['image_id']].append(item['caption'])

    # 合并预测数据与验证数据
    merged_data = []
    for pred_item in pred_data:
        image_id = pred_item['image_id']
        pred_caption = pred_item['caption']

        # 获取真实值 captions
        gt_captions = val_dict.get(image_id, [])

        # 将预测与真实值合并
        merged_data.append({
            "pred": pred_caption,
            "ground_truth": gt_captions
        })

    # 将合并结果保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def build_eval_inputs(coco, coco_res):
    img_ids = coco.getImgIds()
    gts = {}
    res = {}
    for img_id in img_ids:
        gts[img_id] = coco.imgToAnns[img_id]
        res[img_id] = coco_res.imgToAnns[img_id]
    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(gts), tokenizer.tokenize(res)


def eval_single(output_file, annotation_file, total):
    coco = COCO(annotation_file)  # Ground truth JSON file
    coco_res = coco.loadRes(output_file)  # Prediction JSON file

    print('tokenization...')
    gts, res = build_eval_inputs(coco, coco_res)

    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    results = {}
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for metric_name, metric_score in zip(method, score):
                score_percentage = metric_score * 100.0
                print(f"{metric_name}: {score_percentage:.2f}")
                results[metric_name] = score_percentage
        else:
            score_percentage = score * 100.0
            print(f"{method}: {score_percentage:.2f}")
            results[method] = score_percentage

    metrics_to_print = [
        "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
        "METEOR", "ROUGE_L", "CIDEr"
    ]
    average = sum(results[name] for name in metrics_to_print) / len(metrics_to_print)
    print('Samples: {}\nAverage: {:.2f}%\n'.format(total, average))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                total,
                results["Bleu_1"],
                results["Bleu_2"],
                results["Bleu_3"],
                results["Bleu_4"],
                results["METEOR"],
                results["ROUGE_L"],
                results["CIDEr"],
                average,
            ))
    

def process_batch(api_key, batch):
    """
    对一个批次的数据进行评分，返回该批次所有样本的评分列表。
    """
    from openai import OpenAI  # 确保每个子进程加载必要的模块

    client = OpenAI(api_key=api_key, base_url="https://platform.llmprovider.ai/v1")

    message = (
        "Below are the model's predictions and the ground truth answers for a task. "
        "For each case, provide a semantic similarity score between 0 and 10 in the format 'Score: X', "
        "where X is your score. Always use the format 'Score:' and do not explain anything."
        "\n\nResults:\n" +
        "\n".join([f"{i+1}. Pred: {item['pred']}, Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI assistant evaluating a model's prediction quality"}, {"role": "user", "content": message}],
        stream=False
    )

    evaluation_text = response.choices[0].message.content

    # 提取评分
    scores = []
    for line in evaluation_text.splitlines():
        score = float(line.split(":")[1].strip())
        scores.append(score)
    return scores

def deepseek_chat_final(api_key, path, batch_size=10):
    """
    使用多进程评估，返回所有样本的最终平均准确率。
    """
    # 加载 JSON 文件
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分批处理
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")

    # 使用多进程池处理所有批次
    total_score = 0
    total_samples = 0
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )

        # 累计每个批次的评分总和和样本数
        for batch_scores in results:
            total_score += sum(batch_scores)
            total_samples += len(batch_scores)

    # 计算总体评分平均值
    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        output_file, total, truncated_predictions = create_coco_type(
            args.annotation_file,
            args.result_file,
            args.output_dir,
            max_words=args.max_caption_words,
            max_chars=args.max_caption_chars,
        )
        annotation_file, truncated_annotations = normalize_coco_caption_annotation(
            args.annotation_file,
            output_file,
            args.output_dir,
            max_words=args.max_caption_words,
            max_chars=args.max_caption_chars,
        )
        if truncated_predictions > 0 or truncated_annotations > 0:
            print(
                f"Caption sanitization applied: "
                f"{truncated_predictions} predictions, "
                f"{truncated_annotations} reference captions."
            )
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, annotation_file, ans_gt_file)
        eval_single(output_file, annotation_file, total)

        # api_key: use OPENAI_API_KEY from the environment

        # batch_size = 8 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))
