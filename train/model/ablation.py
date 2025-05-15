import os
import json
import argparse
import pandas as pd
from collections import defaultdict
import subprocess
import matplotlib.pyplot as plt

def run_model_variant(variant_name, args):
    """
    运行特定的模型变体，训练和评估
    """
    # 构建目录
    variant_dir = os.path.join(args.result_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    
    # 定义模型特定参数
    train_args = []
    
    if variant_name == "baseline":
        # 基线模型：原始VGG19（冻结）+ 单向LSTM + 元素乘法
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "0.001",
            "--batch_size", "128",
            "--epochs", "10"
        ]
        
    elif variant_name == "vgg_finetune":
        # 只微调VGG19
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "2e-4",
            "--cnn_lr", "2e-5",
            "--batch_size", "128",
            "--epochs", "10"
        ]
        
    elif variant_name == "bi_lstm":
        # 只使用双向LSTM
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "2e-4",
            "--batch_size", "128",
            "--epochs", "10",
            # 此处微调BiLSTM的参数
        ]
        
    elif variant_name == "concat_fusion":
        # 只使用特征级联代替元素乘法
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "2e-4",
            "--batch_size", "128",
            "--epochs", "10",
            # 使用特征级联
        ]
        
    elif variant_name == "full_model":
        # 完整改进模型：VGG微调 + 双向LSTM + 特征级联
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "2e-4",
            "--cnn_lr", "2e-5",
            "--batch_size", "128",
            "--epochs", "10",
            "--focal_loss",
            "--use_swa"
        ]
        
    elif variant_name == "full_model_focal":
        # 完整改进模型 + Focal Loss
        train_args = [
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--save_dir", variant_dir,
            "--lr", "2e-4",
            "--cnn_lr", "2e-5",
            "--batch_size", "128",
            "--epochs", "10",
            "--focal_loss",
            "--use_swa"
        ]
    
    # 执行训练
    print(f"训练变体: {variant_name}")
    train_cmd = ["python", "train.py"] + train_args
    subprocess.run(train_cmd, check=True)
    
    # 执行测试
    best_model_path = os.path.join(variant_dir, "best_acc.pth")
    output_json = os.path.join(variant_dir, "predictions.json")
    
    if os.path.exists(best_model_path):
        # 执行推理
        print(f"测试变体: {variant_name}")
        test_cmd = [
            "python", "test.py",
            "--val_csv", args.val_csv,
            "--question_vocab", args.question_vocab,
            "--answer_vocab", args.answer_vocab,
            "--ckpt", best_model_path,
            "--out_json", output_json
        ]
        subprocess.run(test_cmd, check=True)
        
        # 运行官方评测
        print(f"评测变体: {variant_name}")
        eval_cmd = [
            "python", "../VQA_evaluation/PythonEvaluationTools/vqaEvalDemo.py",
            "--annFile", args.annotation_json,
            "--quesFile", args.question_json,
            "--resFile", output_json
        ]
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        
        # 保存评测结果
        with open(os.path.join(variant_dir, "eval_result.txt"), "w") as f:
            f.write(eval_result.stdout)
            
        # 解析评测结果
        result = parse_evaluation_result(eval_result.stdout)
        return result
    else:
        print(f"错误: 找不到模型检查点 {best_model_path}")
        return None

def parse_evaluation_result(eval_output):
    """
    解析评测输出，提取指标
    """
    result = {}
    
    # 提取总体准确度
    if "Overall Accuracy is" in eval_output:
        acc_line = [line for line in eval_output.split("\n") if "Overall Accuracy is" in line][0]
        result["All"] = float(acc_line.split("is")[1].strip().rstrip("%"))
    
    # 提取各题型准确度
    for qtype in ["Yes/No", "Number", "Other"]:
        if f"Accuracy for {qtype} questions" in eval_output:
            type_line = [line for line in eval_output.split("\n") if f"Accuracy for {qtype} questions" in line][0]
            result[qtype] = float(type_line.split("is")[1].strip().rstrip("%"))
            
    return result

def visualize_results(results, output_path):
    """
    可视化不同模型变体的性能对比
    """
    variants = list(results.keys())
    metrics = ["All", "Yes/No", "Number", "Other"]
    
    # 准备数据
    df = pd.DataFrame(columns=["Variant", "Metric", "Accuracy"])
    for variant in variants:
        for metric in metrics:
            if metric in results[variant]:
                df = df.append({
                    "Variant": variant,
                    "Metric": metric,
                    "Accuracy": results[variant][metric]
                }, ignore_index=True)
    
    # 生成条形图
    plt.figure(figsize=(12, 8))
    
    # 分组条形图
    bar_width = 0.2
    index = range(len(variants))
    
    for i, metric in enumerate(metrics):
        metric_data = df[df["Metric"] == metric]
        plt.bar(
            [x + i * bar_width for x in index], 
            metric_data["Accuracy"], 
            bar_width, 
            label=metric
        )
    
    plt.xlabel('模型变体')
    plt.ylabel('准确率 (%)')
    plt.title('VQA模型变体性能对比')
    plt.xticks([x + bar_width * (len(metrics)-1)/2 for x in index], variants, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300)
    print(f"性能对比图已保存至: {output_path}")
    
    # 生成表格
    table_df = pd.DataFrame(results).T
    table_df.to_csv(output_path.replace(".png", ".csv"))
    print(f"性能对比表已保存至: {output_path.replace('.png', '.csv')}")

def run_ablation_study(args):
    """
    运行消融实验
    """
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 定义要测试的变体
    if args.variants:
        variants = args.variants.split(",")
    else:
        variants = ["baseline", "vgg_finetune", "bi_lstm", "concat_fusion", "full_model", "full_model_focal"]
    
    # 存储结果
    results = {}
    
    for variant in variants:
        print(f"\n{'='*50}\n运行变体: {variant}\n{'='*50}")
        result = run_model_variant(variant, args)
        if result:
            results[variant] = result
    
    # 可视化结果对比
    if results:
        visualize_path = os.path.join(args.result_dir, "ablation_comparison.png")
        visualize_results(results, visualize_path)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQA模型消融实验')
    
    # 数据相关参数
    parser.add_argument('--train_csv', type=str, default='../data_proc/train.csv',
                        help='训练集CSV文件路径')
    parser.add_argument('--val_csv', type=str, default='../data_proc/val.csv',
                        help='验证集CSV文件路径')
    parser.add_argument('--question_vocab', type=str, default='../data_proc/question_vocabs.txt',
                        help='问题词汇表路径')
    parser.add_argument('--answer_vocab', type=str, default='../data_proc/annotation_vocabs.txt',
                        help='答案词汇表路径')
    parser.add_argument('--question_json', type=str, 
                        default='../data_raw/v2_OpenEnded_mscoco_val2014_questions.json',
                        help='问题JSON文件路径')
    parser.add_argument('--annotation_json', type=str,
                        default='../data_raw/v2_mscoco_val2014_annotations.json',
                        help='标注JSON文件路径')
    
    # 实验参数
    parser.add_argument('--result_dir', type=str, default='../ablation_results',
                        help='消融实验结果保存目录')
    parser.add_argument('--variants', type=str, default='',
                        help='要测试的模型变体，逗号分隔。为空则测试所有变体')
    
    args = parser.parse_args()
    run_ablation_study(args) 