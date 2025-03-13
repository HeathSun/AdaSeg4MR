import fiftyone as fo

# 关闭已有的 FiftyOne session，防止冲突
fo.close_app()

# 设置数据路径
data_path = "K:/Year4/FYP/images/val2017"
labels_path = "K:/Year4/FYP/images/annotations/instances_val2017.json"

# 检查数据集是否已经存在，避免重复加载
dataset_name = "coco-val2017"
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=dataset_name
    )

# 启动 FiftyOne 界面并确保连接
session = fo.launch_app(dataset, port=5151)
session.wait()  # 保持连接，防止 Python 进程退出
