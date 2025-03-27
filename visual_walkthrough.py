import os
import cv2
import numpy as np
from bs4 import BeautifulSoup
import re
import glob
from tqdm import tqdm
import shutil
import textwrap
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def extract_data_from_html(html_file):
    """从HTML文件提取问题回答和指标"""
    try:
        # 尝试使用不同的编码打开文件
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试使用其他编码
            with open(html_file, 'r', encoding='latin1') as f:
                content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # 提取问题和回答
        qa_pairs = []
        qa_divs = soup.find_all('div', class_='qa-pair')
        
        # 如果找不到qa-pair类，尝试其他可能的结构
        if not qa_divs:
            questions = soup.find_all('div', class_='question')
            answers = soup.find_all('div', class_='answer')
            for i, q in enumerate(questions):
                if i < len(answers):
                    qa_pairs.append((q.text.strip(), answers[i].text.strip()))
        else:
            for div in qa_divs:
                question = div.find('div', class_='question')
                answer = div.find('div', class_='answer')
                if question and answer:
                    qa_pairs.append((question.text.strip(), answer.text.strip()))
        
        # 如果仍然没有找到QA对，尝试从HTML中提取问题和回答
        if not qa_pairs:
            # 寻找可能包含Q&A的段落
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.text.strip()
                if text.startswith('Q:') or text.startswith('Question:'):
                    # 找到一个问题
                    question = text.split(':', 1)[1].strip()
                    # 查找下一个段落作为可能的答案
                    next_p = p.find_next('p')
                    if next_p and (next_p.text.strip().startswith('A:') or 
                                   next_p.text.strip().startswith('Answer:')):
                        answer = next_p.text.strip().split(':', 1)[1].strip()
                        qa_pairs.append((question, answer))
        
        # 如果还是没有QA对，添加一些默认值以便调试
        if not qa_pairs:
            print(f"警告: 在 {html_file} 中没有找到问题和回答")
            # 添加默认QA对以便测试显示
            qa_pairs = [
                ("What objects do you see?", "I can see a person and a dog in the image."),
                ("Can you describe the scene?", "This appears to be an outdoor scene with objects.")
            ]
        
        print(f"找到 {len(qa_pairs)} 个问答对")
        
        # 提取指标 (IoU等)
        metrics = {}
        metrics_divs = soup.find_all('div', class_='metric')
        
        # 如果找不到metric类，尝试查找其他可能包含指标的元素
        if not metrics_divs:
            # 尝试查找包含"IoU"或其他指标关键词的段落或元素
            iou_elements = soup.find_all(text=re.compile(r'IoU|Accuracy|Precision'))
            for element in iou_elements:
                text = element.string
                if text:
                    match = re.search(r'([A-Za-z\s]+):\s*([\d.]+)', text)
                    if match:
                        metric_name = match.group(1).strip()
                        metric_value = match.group(2)
                        try:
                            metrics[metric_name] = float(metric_value)
                        except ValueError:
                            metrics[metric_name] = metric_value
        else:
            for div in metrics_divs:
                metric_name = div.find('span', class_='metric-name')
                metric_value = div.find('span', class_='metric-value')
                if metric_name and metric_value:
                    try:
                        metrics[metric_name.text.strip()] = float(metric_value.text.strip())
                    except ValueError:
                        metrics[metric_name.text.strip()] = metric_value.text.strip()
        
        return qa_pairs, metrics
    except Exception as e:
        print(f"处理HTML文件时出错: {html_file}, 错误: {e}")
        # 返回空数据而不是抛出异常
        return [], {'Error': 'Failed to parse HTML'}

def draw_text_with_wrapping(draw, text, position, font, fill, max_width):
    """使用自动换行绘制文本"""
    x, y = position
    lines = textwrap.wrap(text, width=max_width)
    
    # 使用getbbox代替getsize获取行高
    try:
        line_height = font.getbbox("A")[3] + 5  # 使用getbbox方法
    except AttributeError:
        # 向后兼容旧版本PIL
        try:
            line_height = font.getsize("A")[1] + 5
        except:
            line_height = 20  # 默认行高
    
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    
    return y  # 返回下一行的y坐标

def create_frame_pil(segmentation_img, qa_pairs, metrics, progress, total):
    """使用PIL创建更美观的视频帧（黑底白字）"""
    # 设置帧大小和背景色
    frame_width, frame_height = 1280, 720
    background_color = (10, 10, 15)  # 深黑色背景
    text_color = (240, 240, 240)  # 接近白色的文本
    accent_color = (41, 128, 185)  # 亮蓝色作为强调色
    highlight_color = (231, 76, 60)  # 红色作为突出色
    
    # 创建黑色背景图像
    frame = Image.new('RGB', (frame_width, frame_height), background_color)
    draw = ImageDraw.Draw(frame)
    
    # 加载字体
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        header_font = ImageFont.truetype("arial.ttf", 22)
        text_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        # 使用默认字体如果找不到Arial
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 加载并调整分割图像大小
    try:
        img = Image.open(segmentation_img)
        # 计算左侧面板尺寸
        left_panel_width = int(frame_width * 0.6)
        left_panel_height = int(frame_height * 0.75)
        
        # 保持纵横比缩放图像
        img.thumbnail((left_panel_width, left_panel_height), Image.LANCZOS)
        
        # 计算图像在左侧面板中的位置
        x_offset = int((left_panel_width - img.width) / 2)
        y_offset = int((left_panel_height - img.height) / 2)
        
        # 粘贴图像到帧
        frame.paste(img, (x_offset, y_offset))
        
        # 在图像周围绘制边框
        draw.rectangle(
            [(x_offset-2, y_offset-2), (x_offset+img.width+2, y_offset+img.height+2)],
            outline=accent_color, 
            width=2
        )
    except Exception as e:
        print(f"Error loading image: {segmentation_img}, error: {e}")
    
    # 添加图像ID和标题
    img_id = os.path.basename(os.path.dirname(segmentation_img))
    draw.text((20, 20), f"Visual Analysis: Image {img_id}", font=title_font, fill=accent_color)
    
    # 画分隔线
    draw.line([(left_panel_width + 20, 10), (left_panel_width + 20, frame_height - 60)], 
              fill=accent_color, width=2)
    
    # 右侧面板 - 问题与回答部分
    right_x = left_panel_width + 40
    y_pos = 60
    
    # 添加问答标题
    draw.text((right_x, y_pos), "Q&A Analysis", font=header_font, fill=accent_color)
    y_pos += 40
    
    # 问题和回答区域最大宽度
    qa_max_width = 35  # 减小字符数以确保适合
    
    # 调试信息，输出找到的QA对
    print(f"显示 {len(qa_pairs)} 个问答对")
    
    # 确保至少有一个QA对可显示
    if not qa_pairs:
        qa_pairs = [("No questions found", "No answers available")]
    
    # 限制显示的QA对数量，以防过多
    qa_pairs = qa_pairs[:3]  # 最多显示3个问答对
    
    # 显示问题和回答，简化显示逻辑
    for i, (question, answer) in enumerate(qa_pairs):
        # 截断过长的问题和回答
        if len(question) > 100:
            question = question[:97] + "..."
        if len(answer) > 150:
            answer = answer[:147] + "..."
        
        # 问题 - 蓝色
        draw.text((right_x, y_pos), f"Q{i+1}:", font=text_font, fill=accent_color)
        y_pos += 25
        
        # 直接使用更简单的文本绘制方法，确保显示
        wrapped_question = textwrap.fill(question, width=qa_max_width)
        draw.text((right_x + 10, y_pos), wrapped_question, font=small_font, fill=text_color)
        y_pos += 10 + wrapped_question.count('\n') * 20
        
        # 回答 - 红色开头
        y_pos += 5
        draw.text((right_x, y_pos), "A:", font=text_font, fill=highlight_color)
        y_pos += 25
        
        # 简化答案显示
        wrapped_answer = textwrap.fill(answer, width=qa_max_width)
        draw.text((right_x + 10, y_pos), wrapped_answer, font=small_font, fill=text_color)
        y_pos += 20 + wrapped_answer.count('\n') * 20
    
    # 右侧面板 - 指标部分
    metrics_x = right_x
    metrics_y = max(y_pos + 20, frame_height // 2 + 40)
    
    # 添加指标标题
    draw.text((metrics_x, metrics_y), "Performance Metrics", font=header_font, fill=accent_color)
    metrics_y += 40
    
    # 绘制指标
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            metric_text = f"{metric_name}: {metric_value:.4f}"
            
            # 绘制指标名称
            draw.text((metrics_x, metrics_y), metric_name, font=text_font, fill=text_color)
            metrics_y += 20
            
            # 绘制指标条背景
            bar_width = 200
            bar_height = 15
            draw.rectangle(
                [(metrics_x, metrics_y), (metrics_x + bar_width, metrics_y + bar_height)],
                fill=(50, 50, 50)
            )
            
            # 计算填充宽度并绘制填充条
            fill_width = int(min(max(metric_value, 0), 1) * bar_width)
            
            # 根据值选择颜色
            if metric_value < 0.3:
                fill_color = (192, 57, 43)  # 红色 - 低值
            elif metric_value < 0.7:
                fill_color = (243, 156, 18)  # 黄色 - 中等值
            else:
                fill_color = (39, 174, 96)  # 绿色 - 高值
                
            draw.rectangle(
                [(metrics_x, metrics_y), (metrics_x + fill_width, metrics_y + bar_height)],
                fill=fill_color
            )
            
            # 绘制百分比值
            percentage = f"{metric_value*100:.1f}%"
            
            # 使用getbbox或getlength代替getsize
            try:
                text_width = text_font.getbbox(percentage)[2]  # 使用getbbox获取宽度
            except AttributeError:
                try:
                    text_width = text_font.getsize(percentage)[0]  # 向后兼容
                except:
                    text_width = len(percentage) * 8  # 简单估算
            
            draw.text(
                (metrics_x + bar_width + 10, metrics_y),
                percentage,
                font=small_font,
                fill=text_color
            )
            
            metrics_y += 30
        else:
            draw.text((metrics_x, metrics_y), f"{metric_name}: {metric_value}", 
                      font=text_font, fill=text_color)
            metrics_y += 25
    
    # 底部进度条
    progress_percent = progress / total
    bar_width = frame_width - 80
    bar_height = 15
    bar_x = 40
    bar_y = frame_height - 40
    
    # 绘制进度条背景
    draw.rectangle(
        [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
        fill=(50, 50, 50)
    )
    
    # 绘制进度条填充
    fill_width = int(progress_percent * bar_width)
    draw.rectangle(
        [(bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height)],
        fill=accent_color
    )
    
    # 添加进度文本
    progress_text = f"Progress: {progress}/{total} ({progress_percent*100:.1f}%)"
    draw.text((bar_x, bar_y - 25), progress_text, font=small_font, fill=text_color)
    
    # 使用抗锯齿转换为OpenCV格式（BGR）
    frame_np = np.array(frame)
    # 转换RGB到BGR
    frame_np = frame_np[:, :, ::-1].copy()
    
    return frame_np

def load_metrics_from_csv():
    """从results.csv加载指标数据"""
    metrics_by_image = {}
    csv_path = "../test_results/results.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return metrics_by_image
    
    try:
        # 尝试不同编码打开CSV文件
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试使用latin1编码
            df = pd.read_csv(csv_path, encoding='latin1')
        
        # 按图像名称分组，提取指标
        for image_name, group in df.groupby('image_name'):
            base_name = os.path.splitext(image_name)[0]
            metrics = {}
            
            # 提取所有可能的指标
            if 'bbox_iou' in df.columns:
                values = group['bbox_iou'].dropna()
                if not values.empty:
                    metrics['Bbox IoU'] = values.iloc[0]
            
            if 'mask_iou' in df.columns:
                values = group['mask_iou'].dropna()
                if not values.empty:
                    metrics['Mask IoU'] = values.iloc[0]
            
            if 'class_accuracy' in df.columns:
                values = group['class_accuracy'].dropna()
                if not values.empty:
                    metrics['Class Accuracy'] = values.iloc[0]
            
            if 'response_time' in df.columns:
                values = group['response_time']
                if not values.empty:
                    metrics['Response Time'] = values.mean()
            
            metrics_by_image[base_name] = metrics
        
        print(f"已从CSV加载 {len(metrics_by_image)} 个图像的指标数据")
        return metrics_by_image
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
        # 如果pandas不可用或出错，尝试使用csv模块
        try:
            import csv
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    image_name = row.get('image_name', '')
                    if not image_name:
                        continue
                    
                    base_name = os.path.splitext(image_name)[0]
                    
                    if base_name not in metrics_by_image:
                        metrics_by_image[base_name] = {}
                    
                    # 提取所有可能的指标
                    if 'bbox_iou' in row and row['bbox_iou']:
                        metrics_by_image[base_name]['Bbox IoU'] = float(row['bbox_iou'])
                    
                    if 'mask_iou' in row and row['mask_iou']:
                        metrics_by_image[base_name]['Mask IoU'] = float(row['mask_iou'])
                    
                    if 'class_accuracy' in row and row['class_accuracy']:
                        metrics_by_image[base_name]['Class Accuracy'] = float(row['class_accuracy'])
                    
                    if 'response_time' in row and row['response_time']:
                        if 'Response Time' not in metrics_by_image[base_name]:
                            metrics_by_image[base_name]['Response Time'] = []
                        metrics_by_image[base_name]['Response Time'].append(float(row['response_time']))
            
            # 计算平均响应时间
            for base_name, metrics in metrics_by_image.items():
                if 'Response Time' in metrics and isinstance(metrics['Response Time'], list):
                    metrics['Response Time'] = sum(metrics['Response Time']) / len(metrics['Response Time'])
            
            print(f"已从CSV加载 {len(metrics_by_image)} 个图像的指标数据")
            return metrics_by_image
        
        except Exception as inner_e:
            print(f"使用备用方法读取CSV文件时也出错: {inner_e}")
            return {}

def create_video(output_dir="../walkthrough_vid", fps=2):
    """创建可视化的演示视频"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件夹
    visualization_dir = "../test_results/visualization"
    image_folders = [d for d in os.listdir(visualization_dir) 
                    if os.path.isdir(os.path.join(visualization_dir, d)) 
                    and re.match(r'\d+', d)]
    
    if not image_folders:
        print("未找到图像文件夹")
        return
    
    image_folders.sort()  # 排序文件夹以确保顺序一致
    total_images = len(image_folders)
    
    # 加载CSV中的指标数据
    csv_metrics = load_metrics_from_csv()
    
    # 设置视频编写器
    output_video_path = os.path.join(output_dir, "walkthrough.mp4")
    frame_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    if not video_writer.isOpened():
        print("无法创建视频文件")
        return
    
    print(f"正在创建演示视频，总共处理 {total_images} 个图像...")
    
    # 跟踪处理成功的图像数量
    success_count = 0
    
    # 逐个处理图像文件夹
    for i, folder in enumerate(tqdm(image_folders)):
        folder_path = os.path.join(visualization_dir, folder)
        
        # 从CSV中查找该图像的指标数据
        metrics_from_csv = csv_metrics.get(folder, {})
        
        # 尝试多种可能的文件名
        segmentation_img = None
        for img_name in ["q1_segmentation.jpg", "segmentation.jpg", "result.jpg", "output.jpg"]:
            img_path = os.path.join(folder_path, img_name)
            if os.path.exists(img_path):
                segmentation_img = img_path
                break
                
        # 尝试多种可能的HTML文件名
        html_file = None
        for html_name in ["report.html", "index.html", "result.html"]:
            html_path = os.path.join(folder_path, html_name)
            if os.path.exists(html_path):
                html_file = html_path
                break
        
        # 如果找不到所需文件，列出文件夹中的所有文件以便调试
        if not segmentation_img or not html_file:
            all_files = os.listdir(folder_path)
            print(f"文件夹 {folder} 中的文件: {all_files}")
            
            # 尝试使用任何jpg/png文件作为图像
            if not segmentation_img:
                for file in all_files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        segmentation_img = os.path.join(folder_path, file)
                        print(f"使用替代图像: {file}")
                        break
            
            # 尝试使用任何html文件
            if not html_file:
                for file in all_files:
                    if file.endswith('.html'):
                        html_file = os.path.join(folder_path, file)
                        print(f"使用替代HTML文件: {file}")
                        break
            
            # 如果仍然找不到所需文件，则跳过
            if not segmentation_img or not html_file:
                print(f"跳过文件夹 {folder}: 缺少所需文件")
                continue
        
        # 从HTML提取数据
        qa_pairs, metrics_from_html = extract_data_from_html(html_file)
        
        # 合并HTML和CSV中的指标数据，优先使用CSV数据
        metrics = {**metrics_from_html, **metrics_from_csv}
        
        # 确保指标值在0-1范围内以便可视化
        for k, v in metrics.items():
            if isinstance(v, float) and (k != 'Response Time') and v > 1:
                metrics[k] = min(v / 100.0, 1.0)  # 如果值大于1且不是时间，则假设是百分比格式
        
        # 如果没有指标数据，添加默认值，避免视频帧为空
        if not metrics:
            metrics = {
                "Bbox IoU": 0.0,
                "Mask IoU": 0.0,
                "Class Accuracy": 0.0,
                "Note": "No metrics available"
            }
        
        # 创建视频帧 - 使用PIL版本以获得更好的视觉效果
        frame = create_frame_pil(segmentation_img, qa_pairs, metrics, i+1, total_images)
        if frame is not None:
            video_writer.write(frame)
            success_count += 1
            
            # 保存当前帧作为预览
            if i == 0 or i == total_images//2 or i == total_images-1:
                preview_path = os.path.join(output_dir, f"preview_{i}.jpg")
                cv2.imwrite(preview_path, frame)
    
    # 释放视频编写器
    video_writer.release()
    print(f"视频已保存至 {output_video_path}")
    print(f"成功处理 {success_count}/{total_images} 个图像")

if __name__ == "__main__":
    create_video()
    print("演示视频创建完成！")
