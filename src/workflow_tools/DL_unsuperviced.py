import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import sys
import pytorch_lightning as pl

sys.path.insert(0, "/root/Soil-Column-Procedures")
# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.cluster import KMeans
from src.workflow_tools import dl_config
from src.API_functions.DL import load_data, log, seed


# SimCLR编码器 (使用预训练 ResNet)
def get_simclr_encoder(embedding_dim = 128):
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)
    projection_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
    )
    return nn.ModuleDict({'backbone':backbone, 'projection_head': projection_head})


from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self, encoder, ):
        super().__init__()

        self.encoder = encoder
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


# 伪标签生成
def generate_pseudo_labels(simclr_encoder, images, device, n_clusters=2, n_init = 10):
    simclr_encoder.eval()
    with torch.no_grad():
        features = simclr_encoder['backbone'](images.to(device))
        features = features.view(features.size(0), -1)
        features = features.cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init=n_init)
    kmeans.fit(features)
    pseudo_labels = kmeans.labels_.reshape((images.shape[0],1,images.shape[2],images.shape[3]))

    pseudo_labels = torch.from_numpy(pseudo_labels).long()
    return pseudo_labels

# 训练 UNet
def train_unet(unet_model, train_loader, pseudo_labels, device, num_epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            labels = pseudo_labels[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = unet_model(images)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch: {epoch}, UNet Loss: {loss.item()}")

if __name__ == '__main__':

    # 1. 超参数
    my_parameters = dl_config.get_parameters()
    device = 'cuda'

    simclr_epochs = 10
    unet_epochs = 10
    simclr_lr = 1e-3
    unet_lr = 1e-3

    # 2. 创建虚拟数据
    labeled_data, labels, unlabeled_data, padding_info, unlabeled_padding_info = dl_config.load_and_preprocess_data()

    # 3. 准备数据加载器
    transform_train, _, _, _ = dl_config.get_transforms(my_parameters['seed'])
    train_dataset = load_data.my_Dataset(
        labeled_data,
        labels,
        padding_info,
        transform=transform_train
    )
    dataloader_train_simclr = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=True, drop_last=True)

    # 4. 创建模型
    encoder = get_simclr_encoder()      # Todo
    model = SimCLRModel()

    semantic_model = dl_config.setup_model()
    if my_parameters['compile']:
        semantic_model = torch.compile(semantic_model).to(device)
    else:
        semantic_model = semantic_model.to(device)

    # 5. 训练 SimCLR
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
    trainer.fit(model, dataloader_train_simclr)

    # 6. 生成伪标签
    pseudo_labels = []
    for batch_idx, images in enumerate(train_loader):
        labels = generate_pseudo_labels(simclr_model.encoder, images, device)
        pseudo_labels.append(labels)

    # 7. 训练 UNet
    print("Start UNet Training")
    train_unet(semantic_model, train_loader, pseudo_labels, device, num_epochs=unet_epochs, lr = unet_lr)
    print("Finish UNet Training")

    print("Training complete!")
