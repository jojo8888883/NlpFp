import os
import json
import argparse
import pandas as pd
from pathlib import Path

def process_dataset(questions_json, annotations_json, images_dir, output_csv):
    """
    处理问题和答案文件，生成CSV格式的数据集
    
    参数:
    questions_json: 问题JSON文件路径
    annotations_json: 答案JSON文件路径
    images_dir: 图像目录路径
    output_csv: 输出CSV文件路径
    """
    print(f"处理数据: {questions_json}")
    
    # 加载问题和答案
    with open(questions_json) as f:
        questions_data = json.load(f)
    
    with open(annotations_json) as f:
        annotations_data = json.load(f)
    
    # 创建答案字典以便快速查找
    answer_dict = {}
    for annotation in annotations_data['annotations']:
        question_id = annotation['question_id']
        answers = [ans['answer'] for ans in annotation['answers']]
        answer_dict[question_id] = answers
    
    # 确定数据集类型（训练/验证）
    subset = 'train' if 'train' in questions_json else 'val'
    image_dir_name = f"{subset}2014"  # 目录名称格式
    
    # 准备数据列表
    data_list = []
    for question in questions_data['questions']:
        question_id = question['question_id']
        image_id = question['image_id']
        image_filename = f"COCO_{image_dir_name}_{image_id:012d}.jpg"
        image_path = os.path.join(images_dir, image_dir_name, image_filename)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            continue
            
        question_text = question['question']
        
        # 获取答案（如果有）
        answers = answer_dict.get(question_id, [])
        
        # 添加到数据列表
        data_list.append({
            'question_id': question_id,
            'image_id': image_id,
            'image_path': image_path,
            'question': question_text,
            'answers': '|'.join(answers)  # 使用|分隔多个答案
        })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data_list)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 保存CSV文件
    df.to_csv(output_csv, index=False)
    print(f"已保存CSV文件: {output_csv}, 包含{len(df)}条记录")
    
    return df

def main(args):
    # 处理训练集
    train_df = process_dataset(
        args.train_questions, 
        args.train_annotations,
        args.images_dir,
        args.output_dir + '/train.csv'
    )
    
    # 处理验证集
    val_df = process_dataset(
        args.val_questions,
        args.val_annotations,
        args.images_dir,
        args.output_dir + '/val.csv'
    )
    
    print(f"处理完成! 训练集: {len(train_df)}条记录, 验证集: {len(val_df)}条记录")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将VQA数据集转换为CSV格式')
    
    parser.add_argument('--train_questions', type=str, 
                        default='data_raw/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='训练集问题JSON文件路径')
    
    parser.add_argument('--val_questions', type=str,
                        default='data_raw/v2_OpenEnded_mscoco_val2014_questions.json',
                        help='验证集问题JSON文件路径')
    
    parser.add_argument('--train_annotations', type=str,
                        default='data_raw/v2_mscoco_train2014_annotations.json',
                        help='训练集答案JSON文件路径')
    
    parser.add_argument('--val_annotations', type=str,
                        default='data_raw/v2_mscoco_val2014_annotations.json',
                        help='验证集答案JSON文件路径')
    
    parser.add_argument('--images_dir', type=str,
                        default='data_proc/images',
                        help='调整大小后的图像目录路径')
    
    parser.add_argument('--output_dir', type=str,
                        default='data_proc',
                        help='输出CSV文件的目录')
    
    args = parser.parse_args()
    main(args) 