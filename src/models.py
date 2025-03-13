class ModelV1(nn.Module):
    def __init__(self, input_dimension: int, hidden_layer_units: int, output_dimension: int):
        super().__init__()

        self.model_architecture = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension, hidden_layer_units),
            nn.ReLU(),
            nn.Linear(hidden_layer_units, output_dimension),
            nn.ReLU(),
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1_score": [],
            "confusion_matrix": []
        }

    def forward(self, input_tensor: torch.Tensor):
        return self.model_architecture(input_tensor)

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        return self.history

# input_dimension = 360 * 363 * 3

# model_1 = ModelV1(
#     input_dimension=input_dimension,
#     hidden_layer_units=10,
#     output_dimension=8,
# ).to(device)

# # Hyperparameters
# epochs = 3
# num_labels = 8
# learning_rate = 0.001

# loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model_1.parameters(), lr=learning_rate)
# accuracy_metric = Accuracy(
#     task="multiclass", num_classes=num_labels, average="macro"
# ).to(device)

model_details = {
    "model_name": model_1.__class__.__name__,
    "learning_rate": learning_rate.__str__(),
    "loss_function": loss_function.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "accuracy_metric": accuracy_metric.__class__.__name__,
    "epochs": epochs.__str__(),
}


class ModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*90*90,
                      out_features=output_shape)
        )
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1_score": [],
            "confusion_matrix": []
            }

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

    def record_metric(self, metric_name: str, value: float):
        """
        Records a metric value into the history.

        Parameters:
        - metric_name (str): The name of the metric (e.g., "loss", "accuracy").
        - value (float): The value of the metric to record.
        """
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        """
        Retrieves the history of a specific metric.

        Parameters:
        - metric_name (str): The name of the metric to retrieve.

        Returns:
        - List[float]: A list of recorded values for the specified metric.
        """
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        """
        Retrieves all recorded metrics in the model's history.

        Returns:
        - Dict[str, List[float]]: A dictionary containing all recorded metrics and their values.
        """
        return self.history

# torch.manual_seed(42)
# model_2 = ModelV2(input_shape=3,
#     hidden_units=10,
#     output_shape=len(class_names)).to(device)
# model_2


class ModelV3(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # Progressive channel increase
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Additional block for deeper feature extraction
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        # Improved classifier with MLP
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*4 * 45 * 45, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape)
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1_score": [],
            "confusion_matrix": []
        }

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x

    def record_metric(self, metric_name: str, value: float):
        """
        Records a metric value into the history.

        Parameters:
        - metric_name (str): The name of the metric (e.g., "loss", "accuracy").
        - value (float): The value of the metric to record.
        """
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        """
        Retrieves the history of a specific metric.

        Parameters:
        - metric_name (str): The name of the metric to retrieve.

        Returns:
        - List[float]: A list of recorded values for the specified metric.
        """
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        """
        Retrieves all recorded metrics in the model's history.

        Returns:
        - Dict[str, List[float]]: A dictionary containing all recorded metrics and their values.
        """
        return self.history

# torch.manual_seed(42)
# model_3 = ModelV3(input_shape=3,
#     hidden_units=10,
#     output_shape=len(class_names)).to(device)
# model_3


class ModelV4(nn.Module):
    def __init__(self, input_channels: int, hidden_units: int, output_classes: int):
        super().__init__()

        # Convolutional feature extractor
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        ## ? added later
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1_score": [],
            "confusion_matrix": []
        }


        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 4, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        # Global Average Pooling and Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4, 256),  # Reduced FC size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x



class ModelResNet18(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.base = models.resnet18(weights=weights)

        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()

    def get_optimizer(self):
        return torch.optim.AdamW([
            {'params': self.base.parameters(), 'lr': 3e-5},
            {'params': self.block.parameters(), 'lr': 8e-4}
        ])

    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x


class TrainerModel5(nn.Module):
    def __init__(self, train_loader, val_loader, test_loader=None, num_classes=8, device='cpu'):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.model = ModelResNet18().to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.loss_fxn = nn.CrossEntropyLoss()

        # Initialize all metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes).to(self.device)
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(self.device)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1_score': [], 'val_f1_score': [],
            "test_loss": [], "test_acc": [],
            "test_precision": [], "test_recall": [],
            "test_f1_score": [], "confusion_matrix": []
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion_matrix_metric.reset()

    def training_step(self, x, y):
        pred = self.model(x)
        loss = self.loss_fxn(pred, y)

        # Calculate all metrics
        acc = self.accuracy(pred, y)
        prec = self.precision(pred, y)
        rec = self.recall(pred, y)
        f1 = self.f1(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, acc, prec, rec, f1

    def validation_step(self, x, y):
        with torch.inference_mode():
            pred = self.model(x)
            loss = self.loss_fxn(pred, y)

            # Calculate all metrics
            acc = self.accuracy(pred, y)
            prec = self.precision(pred, y)
            rec = self.recall(pred, y)
            f1 = self.f1(pred, y)

        return loss, acc, prec, rec, f1

    def process_batch(self, loader, step):
        step_name = step.__name__.replace('_', ' ').capitalize() if hasattr(step, '__name__') else "Processing"

        loss, acc, prec, rec, f1 = 0, 0, 0, 0, 0
        self.reset_metrics()
        for X, y in tqdm(loader, total=len(loader), desc=f"Processing Batch - {step_name}",
                         leave=False, position=2, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
            X, y = X.to(self.device), y.to(self.device)
            l, a, p, r, f = step(X, y)
            loss += l.item()
            acc += a.item()
            prec += p.item()
            rec += r.item()
            f1 += f.item()

        n = len(loader)
        return loss / n, acc / n, prec / n, rec / n, f1 / n

    def train(self, epochs):
        for epoch in tqdm(range(epochs), desc="Overall Progress: Epochs", leave=True,
                          position=0, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
            self.reset_metrics()
            # Training phase
            train_loss, train_acc, train_prec, train_rec, train_f1_score = self.process_batch(
                self.train_loader, self.training_step
            )

            # Validation phase
            val_loss, val_acc, val_prec, val_rec, val_f1_score = self.process_batch(
                self.val_loader, self.validation_step
            )

            # Update history
            metrics = [
                train_loss, val_loss,
                train_acc, val_acc,
                train_prec, val_prec,
                train_rec, val_rec,
                train_f1_score, val_f1_score
            ]

            for item, value in zip(self.history.keys(), metrics):
                self.history[item].append(value)

            print(
                f"[Epoch: {epoch + 1}] "
                f"Train: [loss: {train_loss:.3f} acc: {train_acc:.3f} "
                f"prec: {train_prec:.3f} rec: {train_rec:.3f} f1: {train_f1_score:.3f}] "
                f"Val: [loss: {val_loss:.3f} acc: {val_acc:.3f} "
                f"prec: {val_prec:.3f} rec: {val_rec:.3f} f1: {val_f1_score:.3f}]"
            )
            print(f"\nEpoch {epoch + 1}/{epochs} Performance Report:")
            print(f"└─ [Train] Loss: {train_loss:.4f} | Accuracy: {train_acc * 100:.2f}% | Precision: {train_prec:.2f}")
            print(f"└─ [Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}% | Precision: {val_prec:.2f} | Recall: {val_rec:.2f} | F1-Score: {val_f1_score:.2f}")
        print("Finished training and validation.")

    def test(self):
        """
        Evaluate the model on the test set after training is complete.
        Returns a dictionary with test metrics.
        """
        if self.test_loader is None:
            raise ValueError("Test loader was not provided during initialization")

        self.model.eval()
        self.reset_metrics()

        test_loss = 0
        with torch.inference_mode():
            for X, y in tqdm(self.test_loader, desc=f"Testing Phase",
                             leave=False, position=2, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fxn(pred, y)
                test_loss += loss.item()

                # Calculate all metrics
                self.accuracy(pred, y)
                self.precision(pred, y)
                self.recall(pred, y)
                self.f1(pred, y)
                self.confusion_matrix_metric(pred, y)

        # Calculate average test loss
        test_loss /= len(self.test_loader)

        # Compute final metrics
        test_acc = self.accuracy.compute()
        test_precision = self.precision.compute()
        test_recall = self.recall.compute()
        test_f1 = self.f1.compute()
        confusion_matrix = self.confusion_matrix_metric.compute()

        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc.item())
        self.history["test_precision"].append(test_precision.item())
        self.history["test_recall"].append(test_recall.item())
        self.history["test_f1_score"].append(test_f1.item())
        self.history["confusion_matrix"].append(confusion_matrix.cpu().numpy())
        self.confusion_matrix_metric.reset()
        # Print test results
        print(f"\n[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f} | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1-Score: {test_f1:.2f}")
        print("Finished test evaluation.")


import torch.nn.functional as F

class ModelV6(L.LightningModule):
    def __init__(self, lr=0.001):
        super(ModelV6, self).__init__()

        # Convolutional and Linear layers
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.dense_layer_1 = nn.Linear(16 * 54 * 54, 120)
        self.dense_layer_2 = nn.Linear(120, 84)
        self.dense_layer_3 = nn.Linear(84, 20)
        self.output_layer = nn.Linear(20, len(class_names))

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=len(class_names))
        self.train_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.val_accuracy = Accuracy(task="multiclass", num_classes=len(class_names))
        self.val_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.val_recall = Recall(task="multiclass", num_classes=len(class_names))
        self.val_f1 = F1Score(task="multiclass", num_classes=len(class_names))
        self.test_accuracy = Accuracy(task="multiclass", num_classes=len(class_names))
        self.test_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.test_recall = Recall(task="multiclass", num_classes=len(class_names))
        self.test_f1 = F1Score(task="multiclass", num_classes=len(class_names))

        self.lr=lr

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.dense_layer_1(X))
        X = F.relu(self.dense_layer_2(X))
        X = F.relu(self.dense_layer_3(X))
        X = self.output_layer(X)
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # Calculate and log metrics
        self.log("Train/Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train/Accuracy", self.train_accuracy(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train/Precision", self.train_precision(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # Calculate and log metrics
        self.log("Validation/Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation/Accuracy", self.val_accuracy(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation/Precision", self.val_precision(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation/Recall", self.val_recall(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation/F1", self.val_f1(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        # Calculate and log metrics
        self.log("Test/Loss", loss)
        self.log("Test/Accuracy", self.test_accuracy(y_hat, y))
        self.log("Test/Precision", self.test_precision(y_hat, y))
        self.log("Test/Recall", self.test_recall(y_hat, y))
        self.log("Test/F1", self.test_f1(y_hat, y))
        return loss

    def on_train_start(self):
        # Log the computation graph to TensorBoard
        sample_input = torch.randn(1, 3, 224, 224)
        self.logger.experiment.add_graph(self, sample_input.to(self.device))


### docs: https://docs.ultralytics.com/modes/train/#train-settings

# Load a model
model_v7 = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training, according to the Ultralytics docs)
class_names = train_loader.dataset.classes
num_workers = get_num_workers()

# Hyper parameters
epochs = 100
batch_size=32

# # Train the model
results = model_v7.train(
    data="/content/yolo_blood_cell_dataset",
    project='logs',
    name='ModelV7_yolo',
    epochs=epochs,
    workers=num_workers,
    optimizer='auto',
    seed=RANDOM_SEED,
    cos_lr=True,
    exist_ok=True,
    imgsz=640,
    batch=batch_size,
    classes=class_names,
    save_period=1,
    save_json=True,
    cls=0.9, # Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
    val=True, # Enable validation
    plots=True, # Enable plotting of training and validation metrics
    )


