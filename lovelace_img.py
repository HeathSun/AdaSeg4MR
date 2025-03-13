import os
import json
import random
import shutil
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox, iou
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import subprocess
import time
import re
import csv
from collections import defaultdict

# 导入ada_img的必要函数，不执行start_listening
import importlib.util
spec = importlib.util.spec_from_file_location("ada_img", "ada_img.py")
ada_img = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ada_img)

# 禁用ada_img中的语音输出
ada_img.use_voice_interaction = False

# 测试数据集路径
COCO_VAL_IMAGES = "../images/val2017"
COCO_ANNOTATIONS = "../images/annotations/instances_val2017.json"
TEST_IMAGES_DIR = "../test_images"  # ada_img中默认使用的图片目录

# 测试结果存储路径
RESULTS_DIR = "../test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class AdaImgTester:
    def __init__(self):
        self.coco = COCO(COCO_ANNOTATIONS)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [cat['name'] for cat in self.categories]
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.categories}
        
        # 清空测试图片目录
        if os.path.exists(TEST_IMAGES_DIR):
            shutil.rmtree(TEST_IMAGES_DIR)
        os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
        
        # 创建结果目录结构
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.answers_dir = os.path.join(RESULTS_DIR, "answers")
        self.masks_dir = os.path.join(RESULTS_DIR, "masks") 
        self.boxes_dir = os.path.join(RESULTS_DIR, "boxes")
        self.timing_dir = os.path.join(RESULTS_DIR, "timing")
        
        for dir_path in [self.answers_dir, self.masks_dir, self.boxes_dir, self.timing_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # CSV文件存储综合结果
        self.csv_path = os.path.join(RESULTS_DIR, "results.csv")
        # 创建CSV文件头
        with open(self.csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'image_name', 'question', 'answer', 'response_time', 
                'bbox_iou', 'mask_iou', 'class_accuracy'
            ])
        
        # 测试结果存储
        self.results = defaultdict(dict)
        
        # 跟踪已找到的物体类别
        self.found_objects = set()

    def create_test_subset(self, num_classes=10, num_images_per_class=5):
        """为指定数量的类别创建一个小型测试集（每类num_images_per_class张图片）"""
        print(f"Creating test subset, selecting {num_classes} classes with {num_images_per_class} images each...")
        
        # 随机选择指定数量的类别
        selected_cat_ids = random.sample(self.cat_ids, min(num_classes, len(self.cat_ids)))
        
        test_image_ids = set()
        test_images_by_category = defaultdict(list)
        
        # 对于每个选定类别，选择num_images_per_class张图片
        for cat_id in selected_cat_ids:
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            # 随机选择num_images_per_class张图片
            selected_img_ids = random.sample(img_ids, min(num_images_per_class, len(img_ids)))
            test_image_ids.update(selected_img_ids)
            
            cat_name = next(cat['name'] for cat in self.categories if cat['id'] == cat_id)
            test_images_by_category[cat_name].extend(selected_img_ids)
        
        # 复制选中的图片到测试目录
        test_images_info = {}
        print(f"Selected {len(test_image_ids)} unique images for testing")
        
        for img_id in test_image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            src_path = os.path.join(COCO_VAL_IMAGES, img_info['file_name'])
            dest_path = os.path.join(TEST_IMAGES_DIR, img_info['file_name'])
            
            # 复制图片
            shutil.copy(src_path, dest_path)
            
            # 获取该图片中的所有类别
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            classes_in_image = set()
            for ann in anns:
                cat_id = ann['category_id']
                cat_name = next(cat['name'] for cat in self.categories if cat['id'] == cat_id)
                classes_in_image.add(cat_name)
            
            test_images_info[img_info['file_name']] = {
                'img_id': img_id,
                'classes': list(classes_in_image)
            }
        
        return test_images_info

    def generate_initial_questions(self, classes_in_image):
        """生成前两个测试问题"""
        questions = []
        self.found_objects = set()  # 重置找到的物体集合
        
        # 第一个问题：尽可能找3个物体
        if len(classes_in_image) >= 3:
            selected_classes = random.sample(classes_in_image, 3)
            # 记录将要查找的类别
            self.found_objects.update(selected_classes)
            questions.append(f"find the {selected_classes[0]}, the {selected_classes[1]} and the {selected_classes[2]}")
        elif len(classes_in_image) == 2:
            selected_classes = classes_in_image
            self.found_objects.update(selected_classes)
            questions.append(f"find the {selected_classes[0]} and the {selected_classes[1]}")
        else:
            # 只有一个类别
            class_name = classes_in_image[0]
            self.found_objects.add(class_name)
            questions.append(f"find the {class_name}")
        
        # 第二个问题：计数已找到的物体
        found_objects_list = list(self.found_objects)
        if len(found_objects_list) >= 2:
            questions.append(f"how many {found_objects_list[0]}s and {found_objects_list[1]}s are there?")
        else:
            questions.append(f"how many {found_objects_list[0]}s are there?")
        
        return questions

    def generate_followup_questions(self, counts_by_class):
        """基于前两个问题的结果生成后续问题"""
        questions = []
        
        # 选择一个计数大于0的类别，如果没有，则使用第一个找到的类别
        non_zero_classes = [cls for cls, count in counts_by_class.items() if count > 0]
        
        if non_zero_classes:
            # 选择一个非零计数的类别
            target_class = non_zero_classes[0]
        else:
            # 如果所有类别都是0，使用任何一个找到的类别
            found_objects_list = list(self.found_objects)
            target_class = found_objects_list[0] if found_objects_list else None
        
        # 第三个问题：询问位置（使用单数形式）
        if target_class:
            questions.append(f"where is the {target_class}?")
        else:
            # 防止没有找到任何类别的情况
            questions.append("where is the object?")
        
        # 第四个问题：询问详细信息（同样尽量使用非零类别）
        if target_class:
            questions.append(f"what is the {target_class} like?")
        else:
            # 防止没有找到任何类别的情况
            found_objects_list = list(self.found_objects)
            class_name = found_objects_list[0] if found_objects_list else "object"
            questions.append(f"what is the {class_name} like?")
        
        return questions

    def process_image(self, image_filename, image_info):
        """处理单个图像的所有测试问题，将测试结果保存到文件系统"""
        print(f"\nProcessing image: {image_filename}")
        
        # 将图像加载到ada_img
        img_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        img = cv2.imread(img_path)
        
        # 手动设置ada_img的当前图像
        ada_img.current_image = img
        ada_img.frame_buffer = img.copy()
        ada_img.image_files = [img_path]  # 确保图像文件路径存在
        ada_img.current_image_index = 0
        
        # 获取图像中的类别
        classes_in_image = image_info['classes']
        print(f"Classes in image: {', '.join(classes_in_image)}")
        
        # 生成问题1和问题2
        first_two_questions = self.generate_initial_questions(classes_in_image)
        
        # 创建图像的结果目录
        image_base_name = os.path.splitext(image_filename)[0]
        
        # 创建可视化目录
        visualization_dir = os.path.join(RESULTS_DIR, "visualization", image_base_name)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # 保存原始图像
        original_img_path = os.path.join(self.masks_dir, f"{image_base_name}_original.jpg")
        cv2.imwrite(original_img_path, img)
        cv2.imwrite(os.path.join(visualization_dir, "original.jpg"), img)
        
        # 存储当前测试中找到的物体
        found_classes = {}
        counts_by_class = {}  # 存储每个类别的实例计数
        timing_data = {}
        
        # 处理前两个问题
        all_qa_data = []
        
        # 处理每个问题
        for i, question in enumerate(first_two_questions):
            q_num = i + 1
            q_id = f"{image_base_name}_q{q_num}"
            print(f"Question {q_num}: {question}")
            
            # 记录开始时间
            start_time = time.time()
            
            answer = ""
            bbox_iou = 0.0
            mask_iou = 0.0
            
            # 处理问题1（find）
            if i == 0 and "find" in question:
                # 从问题中提取目标类别
                target_classes = []
                for class_name in self.found_objects:
                    if class_name in question:
                        class_id = ada_img.CLASSES.get(class_name.lower(), None)
                        if class_id is not None:
                            target_classes.append(class_id)
                
                print(f"Target classes: {target_classes}")
                print(f"Looking for: {', '.join(self.found_objects)}")
                
                # 直接调用start_segmentation
                result = ada_img.start_segmentation(question)
                
                # 手动设置segmentation_active为True，确保状态保持
                ada_img.segmentation_active = True
                
                # 获取检测结果并格式化回答
                if hasattr(ada_img, 'current_results') and ada_img.current_results is not None:
                    found_objects = []
                    # 直接从current_results获取检测结果
                    for j, cls in enumerate(ada_img.current_results.boxes.cls):
                        class_id = int(cls)
                        class_names = [name for name, id in ada_img.CLASSES.items() if id == class_id]
                        if class_names:
                            class_name = class_names[0]
                            found_objects.append(class_name)
                    
                    # 构建回答
                    not_found = []
                    found_counts = defaultdict(int)
                    
                    # 统计找到的物体
                    for obj in found_objects:
                        found_counts[obj] += 1
                    
                    # 检查哪些目标物体没有找到
                    for target in self.found_objects:
                        if target not in found_counts:
                            not_found.append(target)
                    
                    # 构建回答字符串
                    found_parts = []
                    for obj, count in found_counts.items():
                        found_parts.append(f"{count} {obj}")
                    
                    if found_parts:
                        answer = "I found " + ", ".join(found_parts)
                    else:
                        answer = "I didn't find any of the requested objects"
                    
                    # 添加未找到的物体信息
                    if not_found:
                        answer += f". I didn't find any {', '.join(not_found)}"
                
                # 使用ada_img的process_segmentation_results来创建可视化
                if hasattr(ada_img, 'current_results') and ada_img.current_results is not None:
                    # 创建可视化
                    overlay = ada_img.process_segmentation_results(ada_img.current_results, img)
                    if overlay is not None:
                        # 合并原图和overlay
                        alpha = 0.4
                        combined_frame = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
                        
                        # 保存到可视化目录
                        vis_path = os.path.join(visualization_dir, f"q{q_num}_segmentation.jpg")
                        cv2.imwrite(vis_path, combined_frame)
                        
                        # 计算mask IoU
                        try:
                            mask_iou = self.calculate_mask_iou(image_info['img_id'], ada_img.current_results)
                        except Exception as e:
                            print(f"Error calculating mask IoU: {e}")
                            mask_iou = 0.0
                        
                        # 计算bbox IoU
                        bbox_metrics = self.calculate_metrics(image_filename, image_info)
                        if 'avg_bbox_iou' in bbox_metrics:
                            bbox_iou = bbox_metrics['avg_bbox_iou']
                
                # 提取并保存边界框信息
                if hasattr(ada_img, 'current_results') and ada_img.current_results is not None:
                    try:
                        boxes_info = []
                        for j, cls in enumerate(ada_img.current_results.boxes.cls):
                            class_id = int(cls)
                            class_names = [name for name, id in ada_img.CLASSES.items() if id == class_id]
                            if class_names:
                                class_name = class_names[0]
                                box = ada_img.current_results.boxes.xyxy[j].cpu().numpy()
                                confidence = float(ada_img.current_results.boxes.conf[j].cpu().numpy())
                                
                                boxes_info.append({
                                    'class_name': class_name,
                                    'box': box.tolist(),
                                    'confidence': confidence
                                })
                                
                                # 更新找到的类别
                                found_classes[class_name] = found_classes.get(class_name, 0) + 1
                    
                        # 保存边界框信息
                        if boxes_info:
                            boxes_path = os.path.join(self.boxes_dir, f"{q_id}_boxes.json")
                            with open(boxes_path, 'w') as f:
                                json.dump(boxes_info, f, indent=2)
                        
                        # 计算边界框IoU
                        bbox_metrics = self.calculate_metrics(image_filename, image_info, boxes_info)
                        if 'avg_bbox_iou' in bbox_metrics:
                            bbox_iou = bbox_metrics['avg_bbox_iou']
                    except Exception as e:
                        print(f"Error extracting bounding box info: {e}")
            
            # 处理问题2（how many）
            elif i == 1 and "how many" in question:
                # 提取类别
                class_names = []
                for obj in self.found_objects:
                    if obj in question:
                        class_names.append(obj)
                
                # 合并答案
                answers = []
                for class_name in class_names:
                    # 调用ada_img的函数
                    count_response = ada_img.get_instance_count(class_name)
                    if isinstance(count_response, str):
                        answers.append(count_response)
                    else:
                        counts_by_class[class_name] = count_response  # 存储计数结果
                        answers.append(f"There are {count_response} {class_name}(s)")
                
                answer = ". ".join(answers)
                print(f"Answer: {answer}")
                
                # 保存答案
                answer_path = os.path.join(self.answers_dir, f"{q_id}_answer.txt")
                with open(answer_path, 'w') as f:
                    f.write(answer)
                
                # 创建带答案的可视化图像
                self.create_answer_visualization(img.copy(), answer, visualization_dir, f"q{q_num}_count")
            
            # 记录结束时间和响应时间
            end_time = time.time()
            response_time = end_time - start_time
            timing_data[f"q{q_num}"] = response_time
            print(f"Response time: {response_time:.6f} seconds")
            
            # 收集QA数据
            qa_data = {
                'question': question,
                'answer': answer,
                'response_time': response_time,
                'bbox_iou': bbox_iou,
                'mask_iou': mask_iou
            }
            all_qa_data.append(qa_data)
            
            # 保存问题和答案
            qa_path = os.path.join(self.timing_dir, f"{q_id}_qa.json")
            with open(qa_path, 'w') as f:
                json.dump(qa_data, f, indent=2)
            
            # 添加CSV条目
            self.add_csv_entry(image_filename, question, answer, response_time, bbox_iou, mask_iou, 0.0)
        
        # 基于前两个问题的结果生成问题3和问题4
        remaining_questions = self.generate_followup_questions(counts_by_class)
        
        # 处理问题3和问题4
        for i, question in enumerate(remaining_questions):
            q_num = i + 3
            q_id = f"{image_base_name}_q{q_num}"
            print(f"Question {q_num}: {question}")
            
            # 记录开始时间
            start_time = time.time()
            
            answer = ""
            # 这些问题不需要IoU
            bbox_iou = None
            mask_iou = None
            
            # 处理位置问题
            if "where is" in question:  # 注意这里改成了"where is"
                # 从问题中提取类别
                for obj in self.found_objects:
                    if obj in question:
                        class_name = obj
                        if class_name.endswith('s'):
                            class_name = class_name[:-1]  # 移除复数形式
                        
                        # 调用ada_img的函数获取位置
                        position_info = ada_img.get_object_positions(class_name)
                        print(f"Answer: {position_info}")
                        
                        # 保存答案
                        answer_path = os.path.join(self.answers_dir, f"{q_id}_answer.txt")
                        with open(answer_path, 'w') as f:
                            f.write(position_info)
                        
                        answer = position_info
                        
                        # 创建带答案的可视化图像
                        self.create_answer_visualization(img.copy(), answer, visualization_dir, f"q{q_num}_location")
                        break
            
            # 处理描述问题
            elif "what is" in question:
                # 从问题中提取类别
                for obj in self.found_objects:
                    if obj in question:
                        class_name = obj
                        
                        # 捕获问题和答案
                        query = f"Describe the {class_name} in the image"
                        print(f"DEBUG: About to call answer_visual_question...")
                        
                        # 保存当前的stdout，用于捕获Ada的输出
                        import io
                        from contextlib import redirect_stdout
                        
                        f = io.StringIO()
                        ada_output = ""
                        
                        try:
                            # 重定向stdout并调用函数
                            with redirect_stdout(f):
                                result = ada_img.answer_visual_question(query)
                            
                            # 获取捕获的输出
                            output = f.getvalue()
                            
                            # 从输出中提取Ada的回答
                            ada_lines = []
                            capture = False
                            for line in output.splitlines():
                                if line.startswith("Ada:"):
                                    capture = True
                                    # 去掉"Ada:"前缀
                                    ada_lines.append(line[4:].strip())
                                elif capture:
                                    ada_lines.append(line.strip())
                            
                            ada_output = "\n".join(ada_lines)
                            
                            # 使用捕获的Ada输出作为答案
                            if ada_output:
                                answer = ada_output
                                print(f"Answer: {answer}")
                            else:
                                answer = f"No description available for the {class_name}"
                                print(f"Answer: {answer}")
                            
                            # 保存答案
                            answer_path = os.path.join(self.answers_dir, f"{q_id}_answer.txt")
                            with open(answer_path, 'w') as f:
                                f.write(answer)
                            
                            # 创建可视化
                            self.create_answer_visualization(img.copy(), answer, visualization_dir, f"q{q_num}_description")
                        except Exception as e:
                            error_msg = f"Error analyzing the {class_name}: {str(e)}"
                            print(error_msg)
                            
                            # 保存错误信息
                            answer_path = os.path.join(self.answers_dir, f"{q_id}_error.txt")
                            with open(answer_path, 'w') as f:
                                f.write(error_msg)
                            
                            answer = error_msg
                            
                            # 创建错误可视化
                            self.create_answer_visualization(img.copy(), error_msg, visualization_dir, f"q{q_num}_error")
                        break
            
            # 记录结束时间和响应时间
            end_time = time.time()
            response_time = end_time - start_time
            timing_data[f"q{q_num}"] = response_time
            print(f"Response time: {response_time:.6f} seconds")
            
            # 收集QA数据
            qa_data = {
                'question': question,
                'answer': answer,
                'response_time': response_time
            }
            # 只有问题1需要记录IoU
            if q_num == 1:
                qa_data['bbox_iou'] = bbox_iou
                qa_data['mask_iou'] = mask_iou
            
            all_qa_data.append(qa_data)
            
            # 保存问题和答案
            qa_path = os.path.join(self.timing_dir, f"{q_id}_qa.json")
            with open(qa_path, 'w') as f:
                json.dump(qa_data, f, indent=2)
            
            # 添加CSV条目 - 只有问题1包含IoU
            if q_num == 1:
                self.add_csv_entry(image_filename, question, answer, response_time, bbox_iou, mask_iou, 0.0)
            else:
                self.add_csv_entry(image_filename, question, answer, response_time, None, None, 0.0)
        
        # 保存响应时间数据
        timing_path = os.path.join(self.timing_dir, f"{image_base_name}_timing.json")
        with open(timing_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        # 创建完整的测试结果HTML页面
        self.create_html_report(image_filename, img, all_qa_data, visualization_dir)
        
        return {
            'image_filename': image_filename,
            'classes_in_image': classes_in_image,
            'found_classes': found_classes,
            'questions': first_two_questions + remaining_questions,
            'timing': timing_data
        }

    def create_answer_visualization(self, img, answer, output_dir, filename_prefix):
        """创建带有问题答案的可视化图像"""
        try:
            # 创建带答案的图像
            h, w = img.shape[:2]
            
            # 添加底部黑色区域用于显示文本
            text_height = 100
            vis_img = np.zeros((h + text_height, w, 3), dtype=np.uint8)
            vis_img[:h, :] = img
            
            # 添加文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)
            line_type = 2
            
            # 将长文本拆分为多行
            max_chars_per_line = 80
            text_lines = []
            
            # 拆分文本
            remaining_text = answer
            while len(remaining_text) > max_chars_per_line:
                # 找到适当的断点
                break_point = remaining_text[:max_chars_per_line].rfind(' ')
                if break_point == -1:
                    break_point = max_chars_per_line
                
                text_lines.append(remaining_text[:break_point])
                remaining_text = remaining_text[break_point:].lstrip()
            
            if remaining_text:
                text_lines.append(remaining_text)
            
            # 绘制文本行
            y_position = h + 20
            for line in text_lines:
                cv2.putText(vis_img, line, (10, y_position), font, font_scale, font_color, line_type)
                y_position += 25
            
            # 保存图像
            output_path = os.path.join(output_dir, f"{filename_prefix}.jpg")
            cv2.imwrite(output_path, vis_img)
        except Exception as e:
            print(f"Error creating answer visualization: {e}")

    def calculate_mask_iou(self, img_id, results):
        """计算掩码IoU，正确处理COCO的多边形分割格式"""
        try:
            # 获取图像的标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            if not anns or not hasattr(results, 'masks') or results.masks is None:
                return 0.0
            
            # 获取图像尺寸
            img_info = self.coco.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']
            
            # 将YOLO的预测掩码转换为正确的尺寸
            pred_masks = []
            for i in range(len(results.masks)):
                # 首先确保tensor在CPU上并转换为numpy数组
                try:
                    # 获取mask的data属性
                    if hasattr(results.masks[i], 'data'):
                        mask_data = results.masks[i].data
                        if hasattr(mask_data, 'cpu'):
                            mask = mask_data.cpu().numpy()
                        else:
                            mask = mask_data
                    else:
                        mask = results.masks[i]
                    
                    if len(mask.shape) == 3:
                        mask = mask[0]  # 如果是3D张量，取第一个通道
                    
                    # 确保掩码尺寸与原图匹配
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.float32), (width, height))
                    
                    # 二值化
                    pred_masks.append((mask > 0.5).astype(np.uint8))
                except Exception as e:
                    print(f"Error processing mask {i}: {str(e)}")
                    continue
            
            # 如果没有成功处理任何掩码，返回0
            if not pred_masks:
                return 0.0
            
            # 计算IoU
            ious = []
            for pred_mask in pred_masks:
                for ann in anns:
                    # 使用COCO API将多边形转换为掩码
                    gt_mask = self.coco.annToMask(ann)
                    
                    # 确保掩码尺寸匹配
                    if gt_mask.shape != pred_mask.shape:
                        gt_mask = cv2.resize(gt_mask.astype(np.float32), (pred_mask.shape[1], pred_mask.shape[0]))
                        gt_mask = (gt_mask > 0.5).astype(np.uint8)
                    
                    # 计算交集和并集
                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        ious.append(iou)
            
            # 返回平均IoU
            return np.mean(ious) if ious else 0.0
        
        except Exception as e:
            print(f"Error calculating mask IoU: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            return 0.0

    def create_html_report(self, image_filename, original_img, qa_data, visualization_dir):
        """创建包含所有测试结果的HTML报告"""
        html_path = os.path.join(visualization_dir, "report.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Results for {image_filename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ display: flex; margin-bottom: 30px; }}
                    .question-section {{ flex: 1; padding: 10px; }}
                    .visualization {{ flex: 2; padding: 10px; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; }}
                    h1, h2 {{ color: #333; }}
                    .metrics {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Test Results for {image_filename}</h1>
                
                <div class="container">
                    <div class="visualization">
                        <h2>Original Image</h2>
                        <img src="original.jpg" alt="Original Image">
                    </div>
                </div>
            """)
            
            # 添加每个问题的数据
            for i, qa in enumerate(qa_data):
                q_num = i + 1
                
                # 确定可视化图像的路径
                vis_path = ""
                if "find" in qa['question']:
                    vis_path = f"q{q_num}_segmentation.jpg"
                elif "how many" in qa['question']:
                    vis_path = f"q{q_num}_count.jpg"
                elif "where is" in qa['question']:
                    vis_path = f"q{q_num}_location.jpg"
                elif "what is" in qa['question']:
                    vis_path = f"q{q_num}_description.jpg"
                
                # 修改这部分，只在Q1显示IoU指标
                metrics_html = f"""
                    <div class="metrics">
                        <p><strong>Response Time:</strong> {qa['response_time']:.6f} seconds</p>
                        {f'<p><strong>Bounding Box IoU:</strong> {qa["bbox_iou"]:.6f}</p>' if q_num == 1 and qa.get("bbox_iou") is not None else ''}
                        {f'<p><strong>Mask IoU:</strong> {qa["mask_iou"]:.6f}</p>' if q_num == 1 and qa.get("mask_iou") is not None else ''}
                    </div>
                """
                
                f.write(f"""
                <div class="container">
                    <div class="question-section">
                        <h2>Question {q_num}</h2>
                        <p><strong>Question:</strong> {qa['question']}</p>
                        <p><strong>Answer:</strong> {qa['answer']}</p>
                        {metrics_html}
                    </div>
                    <div class="visualization">
                        <h2>Visualization</h2>
                        <img src="{vis_path}" alt="Question {q_num} Visualization">
                    </div>
                </div>
                """)
            
            f.write("""
            </body>
            </html>
            """)
        
        print(f"HTML report created at {html_path}")

    def add_csv_entry(self, image_name, question, answer, response_time, bbox_iou, mask_iou, class_accuracy):
        """向CSV文件添加一条记录"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                image_name, question, answer, response_time, 
                bbox_iou, mask_iou, class_accuracy
            ])

    def calculate_metrics(self, image_filename, image_info, boxes_info=None):
        """计算分割结果与真实标注的重合度指标"""
        print(f"Calculating metrics for image {image_filename}...")
        
        img_id = image_info['img_id']
        
        # 获取图像的所有标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 如果没有提供boxes_info，则从文件加载
        if boxes_info is None:
            image_base_name = os.path.splitext(image_filename)[0]
            boxes_files = [f for f in os.listdir(self.boxes_dir) 
                           if f.startswith(image_base_name) and f.endswith('_boxes.json')]
            
            if not boxes_files:
                print(f"No bounding box results found for image {image_filename}")
                return {}
            
            # 使用第一个找到的边界框文件
            boxes_path = os.path.join(self.boxes_dir, boxes_files[0])
            with open(boxes_path, 'r') as f:
                boxes_info = json.load(f)
        
        # 计算IoU (Intersection over Union)
        metrics = {
            'bbox_iou': [],
            'class_accuracy': 0,
            'detection_count': len(boxes_info),
            'ground_truth_count': len(anns)
        }
        
        correct_class_predictions = 0
        
        for box_info in boxes_info:
            pred_class = box_info['class_name']
            pred_box = box_info['box']
            
            # 寻找最佳匹配的真实标注
            best_iou = 0
            best_match_class = None
            
            for ann in anns:
                # 获取真实边界框
                gt_box = list(self.coco.loadAnns(ann['id'])[0]['bbox'])
                # COCO格式为[x, y, width, height]，转换为[x1, y1, x2, y2]
                gt_box[2] += gt_box[0]
                gt_box[3] += gt_box[1]
                
                # 计算IoU
                iou_value = self.calculate_iou(pred_box, gt_box)
                
                if iou_value > best_iou:
                    best_iou = iou_value
                    cat_id = ann['category_id']
                    gt_class = next(cat['name'] for cat in self.categories if cat['id'] == cat_id)
                    best_match_class = gt_class
            
            metrics['bbox_iou'].append(best_iou)
            
            # 检查类别预测是否正确
            if best_match_class == pred_class:
                correct_class_predictions += 1
        
        # 计算类别准确率
        if len(boxes_info) > 0:
            metrics['class_accuracy'] = correct_class_predictions / len(boxes_info)
        
        # 计算平均IoU
        if metrics['bbox_iou']:
            metrics['avg_bbox_iou'] = sum(metrics['bbox_iou']) / len(metrics['bbox_iou'])
        else:
            metrics['avg_bbox_iou'] = 0
        
        return metrics

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # 解析边界框坐标
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算两个边界框的面积
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 计算并集面积
        union_area = box1_area + box2_area - inter_area
        
        # 防止除以零
        if union_area == 0:
            return 0
        
        # 计算IoU
        iou = inter_area / union_area
        return iou

    def run_test(self, num_classes=None, num_images_per_class=None):
        """运行完整的测试过程，支持用户自定义参数"""
        print("Starting Ada image mode test...")
        
        # 如果未指定参数，请求用户输入
        if num_classes is None:
            try:
                num_classes = int(input("Enter number of classes to test (default 10): ") or 10)
            except ValueError:
                num_classes = 10
                print("Invalid input, using default: 10 classes")
        
        if num_images_per_class is None:
            try:
                num_images_per_class = int(input("Enter number of images per class (default 5): ") or 5)
            except ValueError:
                num_images_per_class = 5
                print("Invalid input, using default: 5 images per class")
        
        # 创建测试子集
        test_images_info = self.create_test_subset(num_classes, num_images_per_class)
        
        # 处理每个测试图像
        results = []
        metrics = []
        
        for image_filename, image_info in test_images_info.items():
            # 处理图像
            result = self.process_image(image_filename, image_info)
            results.append(result)
            
            # 计算指标
            metric = self.calculate_metrics(image_filename, image_info)
            metrics.append({
                'image_filename': image_filename,
                **metric
            })
            
            # 更新CSV中的类别准确率
            class_acc = metric.get('class_accuracy', 0.0)
            self.update_csv_class_accuracy(image_filename, class_acc)
        
        # 保存总体结果和指标
        with open(os.path.join(RESULTS_DIR, "test_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 计算总体平均指标
        avg_bbox_iou = sum(m.get('avg_bbox_iou', 0) for m in metrics) / len(metrics) if metrics else 0
        avg_class_accuracy = sum(m.get('class_accuracy', 0) for m in metrics) / len(metrics) if metrics else 0
        
        # 计算掩码IoU的平均值
        mask_ious = []
        for result in results:
            img_filename = result['image_filename']
            base_name = os.path.splitext(img_filename)[0]
            qa_files = [f for f in os.listdir(self.timing_dir) if f.startswith(base_name) and f.endswith('_qa.json')]
            
            for qa_file in qa_files:
                with open(os.path.join(self.timing_dir, qa_file), 'r') as f:
                    qa_data = json.load(f)
                    if 'mask_iou' in qa_data and qa_data['mask_iou'] > 0:
                        mask_ious.append(qa_data['mask_iou'])
        
        avg_mask_iou = sum(mask_ious) / len(mask_ious) if mask_ious else 0
        
        print("\nTest completed!")
        print(f"Processed {len(test_images_info)} images")
        print(f"Average bounding box IoU: {avg_bbox_iou:.4f}")
        print(f"Average mask IoU: {avg_mask_iou:.4f}")
        print(f"Average class accuracy: {avg_class_accuracy:.4f}")
        
        return {
            'results': results,
            'metrics': metrics,
            'avg_bbox_iou': avg_bbox_iou,
            'avg_mask_iou': avg_mask_iou,
            'avg_class_accuracy': avg_class_accuracy
        }
    
    def update_csv_class_accuracy(self, image_filename, class_accuracy):
        """更新CSV中特定图像的类别准确率"""
        # 读取现有CSV
        rows = []
        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # 获取标题行
            for row in reader:
                if row[0] == image_filename:  # 如果是目标图像
                    row[6] = str(class_accuracy)  # 更新类别准确率
                rows.append(row)
        
        # 写回更新后的CSV
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    # 检查必要的文件和目录
    if not os.path.exists(COCO_VAL_IMAGES):
        print(f"Error: COCO validation images directory does not exist: {COCO_VAL_IMAGES}")
        print("Please ensure you have downloaded the COCO dataset and set the paths correctly")
        exit(1)
    
    if not os.path.exists(COCO_ANNOTATIONS):
        print(f"Error: COCO annotations file does not exist: {COCO_ANNOTATIONS}")
        print("Please ensure you have downloaded the COCO dataset annotations")
        exit(1)
    
    # 实例化测试器并运行测试
    tester = AdaImgTester()
    test_results = tester.run_test()
    
    print("\nDetailed results saved to: ", RESULTS_DIR) 