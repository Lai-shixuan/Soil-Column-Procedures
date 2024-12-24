import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
from collections import OrderedDict
from scipy.optimize import curve_fit

class Config:
    """配置参数类"""
    def __init__(self):
        self.mask_threshold: float = 0.8  # 置信度阈值
        self.r_threshold: float = 0.7     # IoU更新阈值
        self.evaluate_interval: int = 5    # 评估间隔
        self.update_interval: int = 1      # 更新间隔
        self.dict_save_scale_factor: float = 1.0  # 字典保存比例因子
        self.Reinit_dict: bool = False    # 是否重新初始化字典
        self.LOG_DIR: str = "./logs"      # 日志目录
        self.should_update: bool = False  # 是否应该更新标签的全局标志
        self.num_classes: int = 2        # 只有背景(0)和目标类别(1)

class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        Args:
            images: 图像数据, shape: (N, H, W)
            labels: 标签数据, shape: (N, H, W)
        """
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.name_list = [f"image_{i}" for i in range(len(images))]
        self.seg_dict = {i: labels[i] for i in range(len(images))}
        self.prev_pred_dict = {}  # 存储模型预测结果
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.name_list[idx]
    
    def update_predictions(self, idx: int, max_probs: torch.Tensor, predictions: torch.Tensor):
        """更新预测结果字典"""
        self.prev_pred_dict[idx] = (max_probs, predictions)
    
    def update_label(self, idx: int, new_label: torch.Tensor):
        """更新单个样本的标签"""
        self.seg_dict[idx] = new_label
        self.labels[idx] = new_label


class SimpleSegNet(nn.Module):
    """简单的分割网络"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, 1)  # 假设二分类问题
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return self.conv2(x)


def update_dataset_labels(IoU_npl_indx: np.ndarray, dataset: CustomDataset, args: Config) -> None:
    """更新数据集中的标签
    Args:
        IoU_npl_indx: 需要更新的类别索引数组
        dataset: 数据集实例
        args: 配置参数
    """
    # 只在更新间隔且有多个类别需要更新时执行
    if epoch % args.update_interval != 0 or len(IoU_npl_indx) <= 1:
        return

    # 遍历数据集中的每个样本
    for idx in range(len(dataset.name_list)):
        # 获取当前图像的分割标签
        seg_label = dataset.seg_dict[idx]
        b, h, w = seg_label.size()

        # 获取模型预测的最大概率值和对应类别
        seg_prediction_max_prob, seg_argmax = dataset.prev_pred_dict[idx]

        # 只更新前景=1的像素，置信度大于阈值且标签不一致的位置
        seg_change_indx = (seg_label != seg_argmax) & (seg_argmax == 1) & (
            seg_prediction_max_prob > args.mask_threshold)

        # 防止过度更新
        seg_label_clone = seg_label.clone()
        seg_label_clone[seg_change_indx] = 1
        if torch.sum(seg_label_clone != 0) < 0.5 * torch.sum(seg_label != 0) and \
           torch.sum(seg_label_clone == 0)/(b*h*w) > 0.95:
            continue

        # 更新标签
        seg_label[seg_change_indx] = 1
        dataset.update_label(idx, seg_label)


class UpdateStrategy:
    """更新策略类：用于确定何时更新标签"""
    
    @staticmethod
    def curve_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """拟合曲线函数
        Args:
            x: 输入数据
            a, b, c: 曲线参数
        Returns:
            np.ndarray: 拟合结果
        """
        return a * (1 - np.exp(-1 / c * x ** b))

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """拟合参数
        Args:
            x: 横坐标数据
            y: 纵坐标数据
        Returns:
            Tuple[float, float, float]: 拟合得到的参数(a, b, c)
        """
        popt, _ = curve_fit(UpdateStrategy.curve_func, x, y, 
                        p0=(1, 1, 1), 
                        method='trf', 
                        sigma=np.geomspace(1, .1, len(y)),
                        absolute_sigma=True, 
                        bounds=([0, 0, 0], [1, 1, np.inf]))
        return tuple(popt)

    @staticmethod
    def derivation(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """计算导数
        Args:
            x: 输入数据
            a, b, c: 曲线参数
        Returns:
            np.ndarray: 导数值
        """
        x = x + 1e-6  # 数值稳定性
        return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))

    @classmethod
    def label_update_epoch(cls, 
                        ydata_fit: np.ndarray, 
                        threshold: float = 0.9, 
                        eval_interval: int = 100, 
                        num_iter_per_epoch: float = 10581 / 10) -> int:
        """确定标签更新的轮次
        Args:
            ydata_fit: IoU历史数据
            threshold: 更新阈值
            eval_interval: 评估间隔
            num_iter_per_epoch: 每轮迭代次数
        Returns:
            int: 建议的更新轮次
        """
        xdata_fit = np.linspace(0, len(ydata_fit) * eval_interval / num_iter_per_epoch, len(ydata_fit))
        a, b, c = cls.fit(xdata_fit, ydata_fit)
        epoch = np.arange(1, 16)
        
        d1 = abs(cls.derivation(epoch, a, b, c))
        d0 = abs(cls.derivation(1, a, b, c))
        relative_change = abs(d1 - d0) / d0
        relative_change[relative_change > 1] = 0
        
        return np.sum(relative_change <= threshold) + 1

    @classmethod
    def if_update(cls, iou_value: np.ndarray, current_epoch: int, threshold: float = 0.90) -> bool:
        """判断是否需要更新
        Args:
            iou_value: IoU历史值数组
            current_epoch: 当前轮次
            threshold: 更新阈值
        Returns:
            bool: 是否应该更新
        """
        update_epoch = cls.label_update_epoch(iou_value, threshold=threshold)
        return current_epoch >= update_epoch

    @classmethod
    def update(cls, IoU_npl_dict: Dict[int, List[float]], args: Config, current_epoch: int) -> np.ndarray:
        """决定是否需要更新标签
        Args:
            IoU_npl_dict: IoU历史值字典
            args: 配置参数
            current_epoch: 当前轮次
        Returns:
            np.ndarray: 需要更新的类别索引数组[0,1]或[0]
        """
        # 背景类永远在更新列表中
        if args.should_update:
            # 如果已经确定要更新，则返回背景和目标类
            return np.array([0, 1])
        
        # 检查目标类是否需要更新
        target_iou = np.array(IoU_npl_dict[1])  # 目标类的IoU历史值
        if cls.if_update(target_iou, current_epoch, threshold=args.r_threshold):
            args.should_update = True  # 设置全局更新标志
            return np.array([0, 1])
        
        return np.array([0])  # 只返回背景类


def evaluate_whole_dataset(IoU_npl_indx: np.ndarray, net: nn.Module) -> Dict[int, float]:
    """评估整个数据集的性能
    Args:
        IoU_npl_indx: 需要评估的类别索引数组
        net: 神经网络模型
    Returns:
        Dict[int, float]: 每个类别的IoU值
    """
    results = {}
    # if only the background class is selected, do not update or eval
    if (epoch % args.evaluate_interval == 0 and len(IoU_npl_indx) > 1):

        if args.Reinit_dict:
            mydataloader.dataset.init_seg_dict()
        # at the end of the epoch, update the dict
        # IoU_npl_indx, which class to update

        net_state_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in net_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.to(torch.device(0))

        eval_net_multiprocess(SpawnContext, net1, IoU_npl_indx, mydataloader, eval_dataloader1, momentum=0, scale_index=args.scale_index, flip=Def_flip, scalefactor=args.dict_save_scale_factor, CRF_post=Def_CRF, tempt_save_root=cfg.LOG_DIR,t_eval=3)
        print('pred_done!')
    return results

def main():
    """主函数"""
    # 配置参数
    args = Config()
    
    # 初始化IoU字典（只需要两个类别）
    IoU_npl_dict: Dict[int, List[float]] = {i: [] for i in range(args.num_classes)}
    
    # 创建示例数据集
    images = np.random.rand(100, 512, 512)  # 100张图片
    labels = np.zeros((100, 512, 512))      # 对应的标签
    
    # 初始化数据集和加载器
    dataset = CustomDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleSegNet().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 主循环
    for epoch in range(100):
        # 训练阶段
        net.train()
        for batch_idx, (imgs, lbls, names) in enumerate(dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = net(imgs.unsqueeze(1))  # 添加通道维度
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            # 更新prev_pred_dict
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(1)
            for i, name in enumerate(names):
                idx = dataset.name_list.index(name)
                dataset.update_predictions(idx, max_probs[i].cpu(), preds[i].cpu())
        
        # 获取需要更新的类别索引
        IoU_npl_indx = UpdateStrategy.update(IoU_npl_dict, args, epoch)
        
        # 评估整个数据集
        iou_results = evaluate_whole_dataset(IoU_npl_indx, net)
        
        # 更新数据集标签
        update_dataset_labels(IoU_npl_indx, dataset, args)
        
        print(f"完成第 {epoch} 轮训练，IoU结果：{iou_results}")
        if args.should_update:
            print("目标类正在进行更新")


if __name__ == "__main__":
    main()