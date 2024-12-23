import numpy as np
from enum import Enum
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import os, cv2, json, shutil, random
from ultralytics import YOLO, YOLOWorld
from ultralytics.data.converter import convert_coco

def coco_to_txt(annotations_path: str, save_dir: str, use_segments=True) -> None:
    """
    将COCO格式的标注文件转换为YOLO格式的txt文件。

    参数:
    - annotations_path (str): COCO格式的标注文件路径。
    - save_dir (str): 保存转换后的txt文件的目录。
    - use_segments (bool): 是否使用多边形分割，默认为True。

    功能:
    1. 如果目标目录已存在，则删除并重新创建。
    2. 使用`convert_coco`函数进行转换，并打印转换完成信息。
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    convert_coco(annotations_path, save_dir=save_dir, use_segments=use_segments, lvis=False, cls91to80=True)
    print('转换完成'.center(50,'-'))

def coco_txt_BaiDu(annotations_path: str, save_dir: str) -> None:
    """
    将COCO格式的JSON标注文件转换为YOLO格式的txt文件（百度版本）。

    参数:
    - annotations_path (str): COCO格式的标注文件路径。
    - save_dir (str): 保存转换后的txt文件的目录。

    功能:
    1. 如果目标目录已存在，则删除并重新创建。
    2. 遍历每个JSON文件，解析图像和标注信息。
    3. 将边界框坐标转换为YOLO格式，并写入对应的txt文件。
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    for file in os.listdir(annotations_path):
        js = json.load(open(os.path.join(annotations_path, file)))
        images = {f'{x["id"]:d}': x for x in js["images"]}
        
        imgToAnns = defaultdict(list)
        for ann in js["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
        
        for img_id, anns in tqdm(imgToAnns.items()):
            img = images[f"{img_id:d}"]
            f = img["file_name"]
            h, w = img["height"], img["width"]
            bboxes = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # Convert to center coordinates
                box[[0, 2]] /= w  # Normalize x coordinates
                box[[1, 3]] /= h  # Normalize y coordinates
                if box[2] <= 0 or box[3] <= 0:  
                    continue
                cls = ann["category_id"]
                bboxes.append([cls] + box.tolist())
            with open(os.path.join(save_dir, str(f).split('.')[0] + '.txt'), '+a') as t:
                for bbox in bboxes:
                    t.write("%g " % bbox[0])
                    for i in bbox[1:]:
                        t.write("%g " % i)
                    t.write("\n")

def spilt_tran_test(li_path: str, train_img_path: str, test_img_path: str,
                     train_label_path: str, test_label_path: str, test_size: float = 0.2,
                    random_state: int = 110, upset_photo: bool = False, verbose=True):
    """
    将数据集划分为训练集和测试集，并将图片和标签复制到指定目录。

    参数:
    - li_path (str): 包含图片和标签的源目录路径。
    - train_img_path (str): 训练集图片保存路径。
    - test_img_path (str): 测试集图片保存路径。
    - train_label_path (str): 训练集标签保存路径。
    - test_label_path (str): 测试集标签保存路径。
    - test_size (float): 测试集比例，默认为0.2。
    - random_state (int): 随机种子，默认为110。
    - upset_photo (bool): 是否重置训练文件，默认为False。
    - verbose (bool): 是否输出详细信息，默认为True。

    功能:
    1. 根据随机种子划分图片和标签为训练集和测试集。
    2. 如果`upset_photo`为True，则删除已有训练文件。
    3. 将训练集和测试集的图片和标签分别复制到指定目录。
    """
    path = Path(li_path)
    random.seed(random_state)
    
    img_path = [str(x) for x in list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg'))]
    label_path = [str(x) for x in list(path.glob('*.txt')) if "class" not in str(x)]
    
    train_img = random.sample(img_path, int(len(img_path) * (1 - test_size)))
    train_label = [x for x in label_path if any(x[:-4] + ext in train_img for ext in ['.jpg', '.png', '.jpeg'])]
    
    test_img = [x for x in img_path if x not in train_img]
    test_label = [x for x in label_path if x not in train_label]
    
    if upset_photo:
        t_img = train_img_path
        v_img = test_img_path

        t_label = train_label_path
        v_label = test_label_path
        
        tree = list(Path(t_img).glob('*.*')) + list(Path(v_img).glob('*.*')) + list(Path(t_label).glob('*.*')) + list(Path(v_label).glob('*.*'))
        for file in tree:
            os.unlink(str(file))
        print('已删除所有训练文件.')

    print('执行复制操作'.center(80,'-'))
    for img in tqdm(train_img):
        if os.path.exists(os.path.join(train_img_path, os.path.basename(img))):
            if verbose: print(img, '\r已存在', end='')
        else:
            shutil.copy2(img, train_img_path)
            if verbose: print(img, '已复制至', train_img_path)
    print("训练图片复制完毕")
    for img in tqdm(test_img):
        if os.path.exists(os.path.join(test_img_path, os.path.basename(img))):
            if verbose: print(img, '\r已存在', test_img_path, end='')
        else:
            shutil.copy2(img, test_img_path)
            if verbose: print(img, '已复制至', test_img_path)
    
    print("测试图片复制完毕")
    for label in tqdm(train_label):
        if os.path.exists(os.path.join(train_label_path, os.path.basename(label))):
            if verbose: print(label, '\r已存在', train_label_path, end='')
        else:
            shutil.copy2(label, train_label_path)
            if verbose: print(label, '已复制至', train_label_path)
    
    print("训练标签复制完毕")
    for label in tqdm(test_label):
        if os.path.exists(os.path.join(test_label_path, os.path.basename(label))):
            if verbose: print(label, '\r已存在', test_label_path, end='')
        else:
            shutil.copy2(label, test_label_path)
            if verbose: print(label, '已复制至', test_label_path)
    
    print("测试标签复制完毕")
    print()
    print('执行完毕'.center(80,'-'))

def train(model_selection: str, yaml_data: str, yolo_world: bool=False, epochs: int = 100, batch: int = -1, val: bool = True,
           save_period: int = -1, project: str = None, pretrained: str = None, single_cls: bool = False,
             lr: float = 0.001, workers: int = 0, seed_change: bool = False, cls: float = 0.5, imgsz: int = 640,
             optimizer="auto", patience=100):
    """
    训练YOLO/YOLOWORLD模型。

    参数:
    - model_selection (str): 模型选择，例如"yolo11n.pt"。
    - yaml_data (str): 数据配置文件路径。
    - yolo_world (bool): 是否使用YOLOWorld模型，默认为False。
    - epochs (int): 训练轮数，默认为100。
    - batch (int): 批次大小，默认为-1（自动选择）。
    - val (bool): 是否进行验证，默认为True。
    - save_period (int): 每隔多少轮保存一次模型，默认为-1（不保存）。
    - project (str): 项目保存路径，默认为None。
    - pretrained (str): 预训练模型路径，默认为None。
    - single_cls (bool): 是否单类别训练，默认为False。
    - lr (float): 学习率，默认为0.001。
    - workers (int): 数据加载线程数，默认为0。
    - seed_change (bool): 是否随机生成种子，默认为False。
    - cls (float): 分类损失权重，默认为0.5。
    - imgsz (int): 图片尺寸，默认为640。
    - optimizer (str): 优化器类型，默认为"auto"。
    - patience (int): 早停耐心值，默认为100。

    功能:
    1. 设置随机种子（如果需要）。
    2. 加载YOLO/YOLOWORLD模型并开始训练。
    3. 打印使用的随机种子。
    """
    seed = random.randint(1, 1e9) if seed_change else 1
    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)
    model.train(data=yaml_data, epochs=epochs, workers=workers, batch=batch,
                save_period=save_period, val=val, pretrained=pretrained,
                project=project, lr0=lr, single_cls=single_cls, imgsz=imgsz,
                  seed=seed, cls=cls, optimizer=optimizer, patience=patience)
    
    print(f'Echo: In this train time, we use the seed of {seed}.')

def eval(model_selection: str, yaml_data: str, yolo_world=False) -> None:
    """
    使用指定的模型和数据集评估模型。

    参数:
    - model_selection (str): 模型选择，用于创建YOLO模型实例。
    - yaml_data (str): 包含数据集的YAML文件路径，或数据集内容的YAML格式。
    - yolo_world (bool): 是否使用YOLOWorld模型，默认为False。

    功能:
    1. 根据模型选择创建YOLO/YOLOWORLD模型实例。
    2. 使用提供的数据集（YAML格式）评估模型。
    3. 打印评估完成信息。
    """
    if yolo_world: YOLOWorld(model_selection).val(data=yaml_data)
    else: YOLO(model_selection).val(data=yaml_data)
    
    print("Evaluation finished.")

class InferDataType(Enum):
    """
    定义了用于推理的数据类型枚举类。

    成员:
    - IMAGE: 表示图像类型的数据。
    - DIR: 表示目录类型的数据。
    """
    IMAGE = "image"
    DIR = "dir"

def get_infer_data(path_dir: str, typing: InferDataType = InferDataType.IMAGE, max_num: int = 5):
    """
    根据指定类型从给定目录中获取推理数据。

    参数:
    - path_dir (str): 目录路径，用于查找推理数据。
    - typing (InferDataType): 指定推理数据的类型，默认为图像类型（InferDataType.IMAGE）。
    - max_num (int): 返回的最大数据数量，默认为5。

    返回:
    - 如果 `typing` 为 `InferDataType.IMAGE`，则返回最多 `max_num` 个随机图像文件路径的列表。
    - 如果 `typing` 为 `InferDataType.DIR`，则返回目录中所有图像文件路径的列表。
    """
    # 获取指定目录下的所有符合条件的图像文件路径
    imgs = [x for x in Path(path_dir).glob('*.*') if "jpg" in str(x) or "png" in str(x) or "jpeg" in str(x)]

    if typing == InferDataType.IMAGE:
        # 对于图像类型，随机抽取最多 max_num 个图像文件路径
        return random.sample(list(imgs), min(max_num, len(imgs)))

    elif typing == InferDataType.DIR:
        # 对于目录类型，返回所有图像文件路径
        return imgs

def predict(model_selection: str, img_path: str, yolo_world=False, conf: float = 0.8, save: bool = False,
                show: bool = True, verbose: bool = True, stream: bool = False):
    """
    使用YOLO/YOLOWORLD模型进行预测。

    参数:
    - model_selection (str): 模型选择，例如"yolo11n.pt"。
    - img_path (str): 输入图片路径。
    - yolo_world (bool): 是否使用YOLOWorld模型，默认为False。
    - conf (float): 置信度阈值，默认为0.8。
    - save (bool): 是否保存结果，默认为False。
    - show (bool): 是否显示结果，默认为True。
    - verbose (bool): 是否输出详细信息，默认为True。
    - stream (bool): 是否流式处理，默认为False。

    功能:
    1. 加载YOLO/YOLOWORLD模型并进行推理。
    2. 显示或保存推理结果。
    """
    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)
    model(source=img_path, conf=conf, show=show, save=save, verbose=verbose, stream=stream)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    """
    主程序入口，用于调用训练、预测或数据预处理函数。
    """
    # 示例调用：
    # predict(r"runs\exp1\train\weights\best.pt", img_path=r"data\images\train\P3_No009.jpg", conf=0.7, verbose=False, stream=False, save=True)
    # coco_to_txt(annotations_path="data/annotations", save_dir="data/new_label", use_segments=True)
    spilt_tran_test("./Annotations/meta/",
                 "data/train/images/", "data/val/images/",
                "data/train/labels/", "data/val/labels/",
                test_size=0.15, random_state=random.randint(1, 1e7), 
                upset_photo=False, verbose=False)

    train(model_selection='./best.pt', yaml_data="./data/data.yaml", yolo_world=True,
      val=True, epochs=100, batch=-1, seed_change=True, imgsz=640, patience=50)
    