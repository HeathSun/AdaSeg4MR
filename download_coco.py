import os
from pathlib import Path
from ultralytics.utils.downloads import download
import zipfile  # To handle unzipping

# 设置数据路径和标注文件路径
data_path = "../images/val2017"  # 你的 val2017 图片目录
annotations_dir = "../images/annotations"  # 设定标注文件的目标目录
annotations_zip = os.path.join(annotations_dir, "annotations_trainval2017.zip")  # 修正路径

# 设置 COCO 图片和标注文件的下载链接
image_urls = [
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
]
annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"  # COCO 标注文件

# 下载 COCO 图片
def download_images():
    # 如果图片目录不存在，则下载
    if not os.path.exists(data_path):
        print("Downloading COCO images...")
        download(image_urls, dir=Path(data_path).parent)
    else:
        print("COCO images already downloaded.")

# 下载 COCO 标注文件
def download_annotations():
    # 如果标注文件不存在，则下载
    if not os.path.exists(annotations_zip):
        print("Downloading COCO annotations...")
        download([annotation_url], dir=annotations_dir)
    else:
        print("COCO annotations already downloaded.")

# 运行下载函数
download_images()
download_annotations()

# 解压文件
def unzip_files():
    # 解压图片文件
    if not os.path.exists(data_path):
        print("Unzipping COCO images...")
        with zipfile.ZipFile(f"../images/val2017.zip", 'r') as zip_ref:
            zip_ref.extractall(Path(data_path).parent)

    # 解压标注文件
    if not os.path.exists(annotations_dir):
        print(f"Unzipping COCO annotations to {annotations_dir}...")
        os.makedirs(annotations_dir, exist_ok=True)  # Ensure annotations directory exists
        if os.path.exists(annotations_zip):
            print(f"Unzipping {annotations_zip}...")
            with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                zip_ref.extractall(annotations_dir)
        else:
            print(f"Annotations file {annotations_zip} not found.")
            return

# 解压文件
unzip_files()

# 使用 FiftyOne 加载数据集
import fiftyone as fo

# 加载 COCO 数据集
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=os.path.join(annotations_dir, "instances_val2017.json"),  # Correct path to the annotations file
    name="coco-val2017"
)

# 启动 FiftyOne 界面
session = fo.launch_app(dataset, port=5151)
