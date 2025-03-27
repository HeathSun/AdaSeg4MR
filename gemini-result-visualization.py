import cv2
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
import re

def decode_mask(mask_str, img_shape):
    """
    解码掩码字符串，将其转换为二进制掩码图像
    注意：这是一个示例实现，需要根据实际的掩码编码格式进行调整
    """
    # 创建空白掩码
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    
    # 示例解码 - 假设<seg_XX>是位置编码
    # 在实际应用中，需要根据确切的编码格式修改此部分
    if "<start_of_mask>" in mask_str:
        try:
            segments = re.findall(r'<seg_(\d+)>', mask_str)
            
            # 简单示例：将每个seg_XX转换为一个填充区域
            # 这里只是一个占位示例，实际实现需要根据真实编码格式
            for i, seg_val in enumerate(segments):
                val = int(seg_val)
                if val > 0:
                    # 在边界框的区域内随机填充一部分区域作为示例
                    # 在实际应用中，这里应该根据真实掩码数据生成正确的掩码
                    mask[val % img_shape[0]:(val % img_shape[0]) + 20, 
                         (val * 2) % img_shape[1]:((val * 2) % img_shape[1]) + 20] = 255
            
        except Exception as e:
            print(f"Error decoding mask: {e}")
    
    return mask

def visualize_results(all_results, img_dir, output_dir=None, specific_image=None):
    """
    可视化Gemini AI结果
    params:
        all_results: 图像名称到JSON数据的映射或列表
        img_dir: 图像目录
        output_dir: 输出目录（如果指定）
        specific_image: 特定图像文件名（如果指定）
    """
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件列表
    if specific_image:
        # 如果指定了特定图像，只处理该图像
        specific_path = Path(img_dir) / specific_image
        img_files = [specific_path] if specific_path.exists() else []
    else:
        # 否则处理目录中的所有图像
        img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
    
    if not img_files:
        print(f"No images found in {img_dir}" + (f" with name {specific_image}" if specific_image else ""))
        return
    
    # 检查all_results的数据类型并准备处理
    is_dict = isinstance(all_results, dict)
    
    # 对于每个图像文件
    for img_path in img_files:
        img_name = img_path.name
        print(f"Processing {img_path}")
        
        # 获取与当前图像对应的数据
        if is_dict:
            # 如果all_results是字典，尝试按图像名称查找
            base_name = os.path.splitext(img_name)[0]
            base_name_without_suffix = base_name.split('_')[0] if '_' in base_name else base_name
            
            # 尝试多种方式查找匹配的数据
            json_data = all_results.get(img_name) or all_results.get(base_name) or \
                       all_results.get(base_name_without_suffix) or []
                       
            if not json_data:
                print(f"No data found for image {img_name}. Tried keys: {img_name}, {base_name}, {base_name_without_suffix}")
                continue
        else:
            # 如果all_results是列表，使用全部数据
            json_data = all_results
            
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
            
        # 创建叠加层
        result = img.copy()
        
        # 为每个检测对象绘制边界框和掩码
        for i, obj in enumerate(json_data):
            # 解析边界框
            x1, y1, x2, y2 = obj["box_2d"]
            
            # 确保边界框在图像范围内
            x1 = max(0, min(x1, img.shape[1]-1))
            y1 = max(0, min(y1, img.shape[0]-1))
            x2 = max(0, min(x2, img.shape[1]-1))
            y2 = max(0, min(y2, img.shape[0]-1))
            
            # 获取标签
            label = obj["label"]
            
            # 为每个对象使用不同的颜色
            np.random.seed(i)  # 使颜色保持一致
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            cv2.putText(result, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 解码并绘制掩码
            if "mask" in obj and obj["mask"]:
                mask = decode_mask(obj["mask"], img.shape)
                
                # 创建彩色掩码
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0] = color
                
                # 将掩码与图像融合
                alpha = 0.5  # 透明度
                cv2.addWeighted(colored_mask, alpha, result, 1, 0, result)
        
        # 保存或显示结果
        if output_dir:
            output_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(output_path, result)
            print(f"Saved result to {output_path}")
        else:
            # 由于可能在无GUI环境中运行，我们默认保存结果而非显示
            output_path = f"result_{img_name}"
            cv2.imwrite(output_path, result)
            print(f"Saved result to {output_path} in current directory")

def load_json_data(json_file):
    """从JSON文件加载数据"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return {}

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Visualize Gemini AI results on images')
    parser.add_argument('--img_dir', default='../gemini_sample_img/', help='Directory containing images')
    parser.add_argument('--output_dir', default='../gemini_results/', help='Directory to save output images')
    parser.add_argument('--json_file', help='JSON file containing detection results (optional)')
    parser.add_argument('--image', help='Specific image file to process (optional)')
    args = parser.parse_args()
    
    # 示例JSON数据 - 按图像名称组织
    sample_data = {
        "000000091615_original.jpg": [
            {"box_2d": [179, 442, 344, 497], "mask": "<start_of_mask><seg_115><seg_7><seg_80><seg_51><seg_51><seg_48><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115>", "label": "the microwave"},
            {"box_2d": [129, 126, 269, 240], "mask": "<start_of_mask><seg_115><seg_115><seg_65><seg_107><seg_115><seg_115><seg_115><seg_7><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115>", "label": "the tv"},
            {"box_2d": [332, 634, 396, 680], "mask": "<start_of_mask><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115><seg_115>", "label": "the cup"}
        ],
        # 可以添加更多图像的数据
        "000000200667": [
            {"box_2d": [464, 497, 680, 664], "mask": "<start_of_mask><seg_115><seg_71><seg_115><seg_65><seg_51><seg_65><seg_51><seg_51><seg_115><seg_115><seg_65><seg_115><seg_115><seg_115><seg_115><seg_115>", "label": "the cow"}
        ]
    }
    
    # 尝试从JSON文件加载数据，如果指定了文件
    data = load_json_data(args.json_file) if args.json_file else sample_data
    
    # 可视化结果
    visualize_results(data, args.img_dir, args.output_dir, args.image)
    
    print("Visualization complete!")