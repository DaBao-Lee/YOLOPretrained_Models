"""
本模块提供了一系列函数，用于处理数据转换、文件操作和模型训练等任务

功能
---
- 将不同格式的标注文件转换为YOLO格式的txt文件
- 移除图像文件中的ICC色彩配置文件
- 标准化标签文件中的标签值
- 检查并复制缺失的文件
- 划分数据集
- 训练和评估YOLO/YOLOWORLD模型
- 导出模型
"""
import numpy as np
from enum import Enum
from lxml import etree
from typing import Union
from tqdm.auto import tqdm
from collections import defaultdict
from ultralytics import YOLO, YOLOWorld
from PIL import Image, ImageDraw, ImageFont
import os, cv2, json, shutil, random, logging, glob
from ultralytics.data.converter import convert_coco

logging.basicConfig(level=logging.INFO)

def coco_to_txt(annotations_path: str, save_dir: str, use_segments=True) -> None:
    """
    将COCO格式的标注文件转换为YOLO格式的txt文件。

    参数:
    - annotations_path (str): COCO格式的标注文件路径。
    - save_dir (str): 保存转换后的txt文件的目录。
    - use_segments (bool): 是否使用多边形分割，默认为True。
    """

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    convert_coco(annotations_path, save_dir=save_dir, use_segments=use_segments, lvis=False, cls91to80=True)
    logging.info('转换完成')


def coco_txt_BaiDu(annotations_path: str, save_dir: str) -> None:
    """
    将COCO格式的JSON标注文件转换为YOLO格式的txt文件（百度版本）。

    参数:
    - annotations_path (str): COCO格式的标注文件路径。
    - save_dir (str): 保存转换后的txt文件的目录。
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
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
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
    logging.info('转换完成')


def voc_to_txt(annotations_path: str, save_dir: str, obj_dict: dict=None) -> None:
    """
    将VOC格式的注释文件转换为YOLO格式的.txt文件。

    参数:
    - annotations_path(str): 注释文件所在的路径。
    - save_dir(str): 转换后的.txt文件保存目录。
    - obj_dict(dict): 包含对象名称与类别编号的字典 e.g {"wheel": 0, "handle": 1, "base": 2}
    """

    assert save_dir.endswith('/') , 'save_dir 必须以 / 结尾'
    assert obj_dict is not None, '请输入对象名称与类别编号的字典'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    xmls = glob.glob(os.path.join(annotations_path, '*.xml'))
    for xml_name in tqdm(xmls):
        txt_name = os.path.basename(xml_name).replace('xml', 'txt')
        f = open(os.path.join(save_dir, txt_name), '+w')
        with open(xml_name, 'rb') as fp:
            xml = etree.HTML(fp.read())
            width = int(xml.xpath('//size/width/text()')[0])
            height = int(xml.xpath('//size/height/text()')[0])
            
            obj = xml.xpath('//object')
            for each in obj:
                name = each.xpath("./name/text()")[0]
                classes = obj_dict[name]
                xmin = int(each.xpath('./bndbox/xmin/text()')[0])
                xmax = int(each.xpath('./bndbox/xmax/text()')[0])
                ymin = int(each.xpath('./bndbox/ymin/text()')[0])
                ymax = int(each.xpath('./bndbox/ymax/text()')[0])
                
                dw = 1 / width
                dh = 1 / height
                x_center = (xmin + xmax) / 2
                y_center = (ymax + ymin) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)
                x, y, w, h = x_center * dw, y_center * dh, w * dw, h * dh
               
                f.write(str(classes) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + '\n')
            f.close()
    logging.info('转换完成')


def rm_icc_profile(src_path: str) -> None:
    """
    从指定路径中的所有PNG图像文件中移除ICC色彩配置文件。

    参数: src_path (str): 包含PNG图像文件的目录路径。
    """

    src_path = glob.glob(os.path.join(src_path, '*.png'))
    for path in tqdm(src_path):
        img = Image.open(path)
        img.save(path, format="PNG", icc_profile=None)
    
    logging.info('icc_profile 已删除')


def normalize_labels(src_path: str) -> None:
    """
    标准化标签文件中的标签值，确保所有值不超过1.0。
    
    参数:
    - src_path (str): 包含标签文件的源路径。
    """

    src_path = glob.glob(os.path.join(src_path, '*.txt'))
    for path in tqdm(src_path):
        f = open(path, "r")
        texts = f.readlines()
        f.close()
        with open(path, "+w") as t:
            for text in texts:
                text = text.split()
                for tt in text[1:]:
                    if float(tt) > 1.0:
                        text[text.index(tt)] = '1.0'
                t.write(text[0] + " ")
                for tt in text[1:]:
                    t.write("%g " % float(tt))
                t.write("\n")
    
    logging.info('Label files normalized over...')


def check_and_copy_missing_files(img_path: str, label_path: str = None,
                                  meta_path: str = None, missing_dir: str = None) -> None:
    """
    检查图像文件和标签文件的对应关系，并可选地检查元数据文件。
    
    参数:
    img_path (str): 图像文件夹路径。
    label_path (str): 标签文件夹路径。
    meta_path (str, 可选): 元数据文件夹路径，默认为None。
    """
    
    imgs_name = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_path, "*.*")) if x.endswith((".jpg", ".png", "jpeg"))]

    if label_path is not None:
        labels_name = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(label_path, "*.txt"))]
        if len(imgs_name) >= len(labels_name):
            for name in tqdm(imgs_name):
                if name not in labels_name:
                    print(f"{name} not in labels")
                    if missing_dir:
                        src_img = next(glob.glob(os.path.join(img_path, f"{name}.*")))
                        shutil.copy2(src_img, missing_dir)
        else:
            for name in tqdm(labels_name):
                if name not in imgs_name:
                    print(f"{name} not in imgs")
                    if missing_dir:
                        src_label = next(glob.glob(os.path.join(label_path, f"{name}.txt")))
                        shutil.copy2(src_label, missing_dir)
    else:
        logging.warning("labels_name is None")

    if meta_path is not None:
        meta_name = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(meta_path, "*.*")) if x.endswith((".jpg", ".png", "jpeg"))]
        for name in tqdm(meta_name):
            if name not in imgs_name:
                print(f"{name} not in imgs")
                if missing_dir:
                    src_meta = next(glob.glob(os.path.join(meta_path, f"{name}.*")))
                    shutil.copy2(src_meta, missing_dir)

    logging.info('检查完成...')

def rm_files(src_files: str, target_file: str) -> None:
    """
    删除指定目录中的目标文件及其对应的txt文件。

    参数:
    - src_files (str): 源文件目录路径。
    - target_file (str): 目标文件名（不带扩展名）。
    """
    
    assert os.path.exists(src_files), f'{src_files} 不存在'
    files = glob.glob(os.path.join(src_files, '*.*'))

    target_file_base = os.path.splitext(target_file)[0]
    target_txt = os.path.join(src_files, target_file_base + ".txt")
    target_img = [os.path.join(src_files, target_file_base + ext) for ext in ['.jpg', '.png', '.jpeg']]

    if target_txt in files:
        os.remove(target_txt)
        logging.info(f"{target_txt} 已删除")
    else: logging.warning(f"{target_txt} 不存在")

    for img in target_img:
        if img in files:
            os.remove(img)
            logging.info(f"{img} 已删除")

def split_meta(meta_path: str, split_name: str="meta", split_num: int=5):
    """
    根据给定的路径分割元数据。

    该函数会将指定路径下的图片和标签文件分割成若干部分，每部分包含大致相等数量的图片和标签。
    
    参数:
    - meta_path: 包含图片和标签文件的路径。
    - split_name: 分割后的文件夹名称前缀。
    - split_num: 分割的数量。
    """
    
    imgs = []
    for ext in ["jpg", "png", "jpeg"]:
        imgs.extend(glob.glob(meta_path.rstrip('/') + f"/*.{ext}"))
    labels = glob.glob(meta_path.rstrip('/') + "/*.txt")
    
    if len(imgs) != len(labels): 
        logging.error("imgs and labels not match")
        return 

    length = len(imgs) // split_num + 1
    if os.path.exists('./meta_split_results'):
        shutil.rmtree('./meta_split_results')
    os.mkdir('./meta_split_results')
    for i in range(split_num):
        os.mkdir(f'./meta_split_results/{split_name}{i}')
        for index in range(i*length, length * (i + 1)):
            try:
                shutil.copy2(imgs[index], f'./meta_split_results/{split_name}{i}')
                shutil.copy2(labels[index], f'./meta_split_results/{split_name}{i}')
            except: pass

    logging.info("split done...")

def collcet_meta(meta_split_path: str="./meta_split_results", save: bool=False) -> list:
    """
    该函数会查找并收集指定目录中所有子目录下的元数据文件路径。
    如果指定的目录不存在，则记录错误信息并返回。

    参数:
    - meta_split_path (str): 包含分割结果的目录路径，默认为"./meta_split_results"。
    - save (bool): 是否保存收集到的元数据文件，默认为False。

    返回:
    list: 包含所有收集到的元数据文件路径的列表。
    """
    if not os.path.exists(meta_split_path):
        logging.error("no split results found")
        return

    split_results = glob.glob(f"{meta_split_path.rstrip('/')}/*")
    meta = []
    for split_result in split_results:
        meta.extend(glob.glob(split_result.rstrip('/') + "/*.*"))
    
    logging.info("collect done...")

    if save:
        os.mkdir('./meta')
        copy_files(meta, './meta', verbose=False)
        logging.info("meta files saved at ./meta")

    return meta

def copy_files(src_files: Union[list, str], dest_dir: str, verbose: bool = True, max_num:int=None) -> None:
    """
    将文件从源路径复制到目标目录。

    参数:
    - src_files (list|str): 源文件路径列表。
    - dest_dir (str): 目标目录路径。
    - verbose (bool): 是否输出详细信息，默认为True。
    - max_num (int): 随机选择指定数量的目标文件进行复制，默认为None。
    """

    if isinstance(src_files, str):
        if src_files.startswith('.'): logging.warning("src_files以 . 开头可能会导致错误发生!!!")
        src_files = glob.glob(os.path.join(src_files, '*.*'))
        if len(src_files) == 0: 
            logging.warning("src_files为空, 请检查路径是否正确或者去掉源文件路径的.开头")
            return
    
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    if max_num is not None:
        src_files = random.sample(src_files, min(max_num, len(src_files)))
        logging.info(f'确定将{max_num}个目标文件复制到{dest_dir}?[y/n]')
        y_n = input()
        if y_n.lower() != 'y': 
            logging.info("取消复制操作")
            return 

    for file in tqdm(src_files):
        dest_path = os.path.join(dest_dir, os.path.basename(file))
        if os.path.exists(dest_path):
            if verbose:
                logging.info(f"{file} 已存在")
        else:
            shutil.copy2(file, dest_dir)
            if verbose:
                logging.info(f"{file} 已复制至 {dest_dir}")


def spilt_train_test(li_path: Union[str, list], train_img_path: str, test_img_path: str,
                     train_label_path: str, test_label_path: str, negative_path: str = None,
                     test_size: Union[int, float] = .2, random_state: int = 110,
                     upset_photo: bool = False, verbose=True):
    """
    将数据集划分为训练集和测试集，并将图片和标签复制到指定目录。

    参数:
    - li_path (str): 包含图片和标签的源目录路径。
    - train_img_path (str): 训练集图片保存路径。
    - test_img_path (str): 测试集图片保存路径。
    - train_label_path (str): 训练集标签保存路径。
    - test_label_path (str): 测试集标签保存路径。
    - test_size (float): 测试集比例，默认为None。
    - random_state (int): 随机种子，默认为110。
    - upset_photo (bool): 是否重置训练文件，默认为False。
    - verbose (bool): 是否输出详细信息，默认为True。
    - negative_path (str): 负样本图片路径，默认为None。
    """

    path = li_path
    random.seed(random_state)
    
    if isinstance(li_path, str):
        img_path = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpeg'))
        label_path = [x for x in glob.glob(os.path.join(path, '*.txt')) if "class" not in x]
    
    if isinstance(li_path, list):
        img_path = [x for x in li_path if x.endswith(('.jpg', '.png', '.jpeg'))]
        label_path = [x for x in li_path if x.endswith('.txt')]
        
    if isinstance(test_size, float):
        test_img = random.sample(img_path, int(len(img_path) * test_size))
    else:
        test_img = random.sample(img_path, test_size)
    
    test_label_names = [x.rsplit('/')[-1].split('.')[0] for x in test_img]
    test_label = [x for x in label_path if x.rsplit('/')[-1].split('.')[0] in test_label_names]
    train_img = [x for x in img_path if x not in test_img]

    if negative_path is not None:
        train_img += random.sample(glob.glob(os.path.join(negative_path, '*.jpg')), min(2500, int(len(train_img) * .1)))

    train_label = [x for x in label_path if x not in test_label]
    
    if upset_photo:
        t_img = train_img_path
        v_img = test_img_path

        t_label = train_label_path
        v_label = test_label_path
        
        tree = glob.glob(os.path.join(t_img, '*.*')) + glob.glob(os.path.join(v_img, '*.*')) + glob.glob(os.path.join(t_label, '*.*')) + glob.glob(os.path.join(v_label, '*.*'))
        for file in tree:
            os.unlink(file)
        logging.info('已删除所有训练文件.')

    copy_files(train_img, train_img_path, verbose)
    logging.info("训练图片复制完毕")
    
    copy_files(test_img, test_img_path, verbose)
    logging.info("测试图片复制完毕")
    
    copy_files(train_label, train_label_path, verbose)
    logging.info("训练标签复制完毕")
    
    copy_files(test_label, test_label_path, verbose)
    logging.info("测试标签复制完毕")
    
    logging.info('执行完毕')


def train(model_selection: str, yaml_data: str, yolo_world: bool=False, epochs: int = 100, batch: int = -1, val: bool = True,
           save_period: int = -1, project: str = None, pretrained: str = None, single_cls: bool = False,
             lr: float = 0.001, workers: int = 0, seed_change: bool = False, cls: float = 0.5, imgsz: int = 640,
             optimizer="SGD", patience=100, resume: bool = False, plots=True, cos_lr=True):
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
    - optimizer (str): 优化器类型，默认为"SGD"。
    - patience (int): 早停耐心值，默认为100。
    - resume (bool): 是否恢复训练，默认为False。
    - plots (bool): 是否绘制训练曲线，默认为True。
    - cos_lr (bool): 是否使用余弦退火学习率，默认为True。
    """
    
    seed = random.randint(1, 1e9) if seed_change else 1
    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)
    model.train(data=yaml_data, epochs=epochs, workers=workers, batch=batch,
                save_period=save_period, val=val, pretrained=pretrained,
                project=project, lr0=lr, single_cls=single_cls, imgsz=imgsz,
                  seed=seed, cls=cls, optimizer=optimizer, patience=patience,
                  resume=resume, plots=plots, cos_lr=cos_lr)
    
    logging.info(f'Echo: In this train time, we use the seed of {seed}.')


def eval(model_selection: str, yaml_data: str, yolo_world=False) -> any:
    """
    使用指定的模型和数据集评估模型。

    参数:
    - model_selection (str): 模型选择，用于创建YOLO模型实例。
    - yaml_data (str): 包含数据集的YAML文件路径，或数据集内容的YAML格式。
    - yolo_world (bool): 是否使用YOLOWorld模型，默认为False。
    """
    
    if yolo_world: result = YOLOWorld(model_selection).val(data=yaml_data)
    else: result = YOLO(model_selection).val(data=yaml_data)
    
    logging.info("Evaluation finished.")

    return result


def get_best_last_model(model_path: str = "./runs/detect/train/weights", index: int = None, mode: int = 0) -> str:
    """
    获取最佳或最后一个模型的路径。

    根据提供的模型路径和模式，本函数会选择并返回最佳模型或最后一个模型的路径。
    如果指定了索引，则会根据索引选择特定训练的模型路径。

    参数:
    - model_path (str): 模型文件所在的目录路径。默认路径为 "./runs/detect/train/weights"。
    - index (int): 训练的索引，如果提供，则会根据索引选择特定的训练路径。默认为 None。
    - mode (int): 选择模式。0 代表选择最佳模型，1 代表选择最后一个模型。默认为 0。

    返回:
    - str: 选定模型的路径。
    """

    if index is not None:
        model_path = f"./runs/detect/train{index}/weights"

    models = glob.glob(os.path.join(model_path, '*.pt'))
    assert models, f"{model_path} 未找到模型，请检查路径是否正确。"

    if mode == 0:
        path = next(x for x in models if "best" in os.path.basename(x))
        logging.info(f"选取最佳模型: {path}")

    elif mode == 1:
        path = next(x for x in models if "last" in os.path.basename(x))
        logging.info(f"选取最后一个模型: {path}")

    logging.info("Model fetched successfully.")

    return str(path)


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

    imgs = [x for x in glob.glob(os.path.join(path_dir, '*.*')) if "jpg" in x or "png" in x or "jpeg" in x]

    if typing == InferDataType.IMAGE:
        return random.sample(list(imgs), min(max_num, len(imgs)))

    elif typing == InferDataType.DIR:
        return imgs


def predict(model_selection: str, img_path: str, yolo_world=False, conf: float = 0.5, save: bool = False,
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
    """

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)
    model(source=img_path, conf=conf, show=show, save=save, verbose=verbose, stream=stream)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_pillow_text(img_array:np.array, font_path: str, text: str, position, textColor=(255, 255, 255), textSize: int = 20):
    """
    在给定的图像数组上添加文本。

    参数：
    - img_array: np.array - 输入图像数组。
    - font_path: str - 字体文件路径。
    - text: str - 要添加到图像上的文本。
    - position: tuple - 文本在图像上的位置（左上角）。
    - textColor: tuple - 文本颜色（默认为白色）。
    - textSize: int - 文本大小（默认为20）。

    返回：
    - 添加文本后的图像数组。
    """

    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        font_path, textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def advanced_predict(model_selection: str, img_path: Union[str, list], yolo_world=False, conf: float = 0.5,
                    save: bool = False, show: bool = True, font_path=None, replace_text: dict = None):
    """
    使用YOLO模型进行高级预测。

    参数:
    - model_selection (str): 模型选择字符串，用于指定使用的YOLO模型。
    - img_path (Union[str, list]): 图像路径或路径列表，表示要进行预测的图像。
    - yolo_world (bool, optional): 是否使用YOLOWorld模型，默认为False。
    - conf (float, optional): 置信度阈值，默认为0.5。
    - save (bool, optional): 是否保存预测结果图像，默认为False。
    - show (bool, optional): 是否显示预测结果图像，默认为True。
    - font_path (str, optional): 字体文件路径，用于在图像上添加文本。
    - replace_text (dict, optional): 替换文本的字典，用于替换检测到的对象名称。
    """

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)

    if isinstance(img_path, str) and not img_path.endswith(('.jpg', '.png', '.jpeg')):
        img_path = [file for x in ['jpg', 'png', 'jpeg'] for file in glob.glob(os.path.join(img_path, f'*.{x}'))]
        
    results = model(img_path, conf=conf, show=False, save=False, verbose=False)
    for result in results:
        names = result.names
        if replace_text is not None:
            names = replace_text
        boxes = result.boxes
        orig_img = result.orig_img
        path = result.path

        for index in range(len(boxes)):
            cls_ = names[boxes[index].cls.cpu().item()]
            x1, y1, x2, y2 = np.int_(boxes.xyxy[index].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (220, 110, 10), 3)
            text_x = max(0, x1)
            text_y = y1
            
            text_bg_color = (220, 110, 10)
            cv2.rectangle(orig_img, (text_x, text_y - 1), (text_x + 50, text_y + 23), text_bg_color, -1)
            if font_path is not None:
                orig_img = add_pillow_text(orig_img, font_path, cls_, (text_x, text_y), textSize=22)
            else: cv2.putText(orig_img, cls_, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if show:
            cv2.imshow('result', orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            if not os.path.exists('results'):
                os.mkdir('./results')
            cv2.imwrite("results/" + path.split('\\')[-1], orig_img)

    logging.info("Prediction finished.")

def export_model(model_selection: str, yolo_world: bool=False, format: str="onnx"):
    """
    根据模型选择和是否为YOLOWorld模型来创建相应的模型实例，并将其导出为指定的格式。

    参数:
    - model_selection (str): 模型选择字符串，用于指定所需的模型类型。
    - yolo_world (bool): 默认为False。指示是否使用YOLOWorld模型的布尔值。
    - format (str): 默认为"onnx"。
    """

    if yolo_world: model = YOLOWorld(model_selection)
    else: model = YOLO(model_selection)

    model.export(format=format)

    logging.info("Model exported successfully.")


if __name__ == '__main__':
    
    meta = collcet_meta(meta_split_path='./meta_split_results')

    spilt_train_test(meta, "data/train/images/", "data/val/images/",
                "data/train/labels/", "data/val/labels/",
                negative_path='./Negative/', test_size=.2,
                random_state=110, upset_photo=True, verbose=False)

    train(model_selection='yolo11n.pt', yaml_data="./data/data.yaml",
     yolo_world=False, val=True, epochs=200, batch=108, seed_change=False,
     imgsz=320, resume=False, single_cls=True, optimizer="SGD", patience=50)
