import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
from typing import Any, Union
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from peak_detector.tlm.utils.evaluator import TLMEvaluator
from peak_detector.tlm.model import TLModel


class TLMTrainer:
    def __init__(
        self,
        model: TLModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        save_freq: int,
        eval_freq: int,
        threshold: float,
    ) -> None:
        """Load in the model and the data, and set up tools needed for training

        Args:
            model (TLModel): transfer learning model with ResNet backbone and MaskRCNN/FastRCNN heads
            train_loader (DataLoader): dataset and labels for training
            test_loader (DataLoader): dataset and labels for testing / evaluation
            num_epochs (int): maximum number of epochs to train for
            save_freq (int): number of epochs after which model weights are saved
            eval_freq (int): number of epochs after which the model performance is evaluated
            threshold (float): confidence threshold for accepting a prediction during evaluation
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.threshold = threshold
        self.last_save_no = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimiser = self.create_optimiser(0.001)
        self.lr_scheduler = self.create_lr_scheduler()
        self.avg_train_losses = []
        self.avg_test_losses = []
        self.mean_ious = []
        self.folder = "peak_detector/tlm/training_outputs"

    def train(self, min_delta: float) -> None:
        """_summary_

        Args:
            min_delta (float): minimum change in loss between epochs to consider the model converged
        """
        start_time = time.perf_counter()
        self.epoch_times = [start_time]
        self.train_losses = []
        self.test_losses = []

        # force 1 based counting so save/evaluation frequencies work
        for epoch in range(1, self.num_epochs + 1):
            print(f"Training epoch: {epoch}")
            self.train_one_epoch(epoch)
            self.calc_test_losses()

            if epoch == self.num_epochs:
                self.histogram(
                    f"{self.folder}/{self.last_save_no}.torch", "final_histogram"
                )
            elif epoch % self.eval_freq == 0:
                if self.last_save_no != 0:
                    print("Evaluating...")
                    self.evaluate(f"{self.folder}/{self.last_save_no}.torch")

            self.epoch_times.append(time.perf_counter())
            print(f"Total time elapsed: {self.epoch_times[-1] - self.epoch_times[0]}")
            # force at least 5 epochs of training before beginning to check for early stopping
            if epoch > 5:
                if self.early_stopping(min_delta) and epoch % self.save_freq != 0:
                    # save model weights and evaluate if the training stops early
                    print("Early stop criteria reached.")
                    self.histogram(
                        f"{self.folder}/{self.last_save_no}.torch", "final_histogram"
                    )
                    filename = f"{self.folder}/{epoch}.torch"
                    torch.save(self.model.state_dict(), filename)
                    print(f"State saved at epoch {epoch}")
                    break

    @torch.no_grad()
    def evaluate(self, state_dict_path: str) -> None:
        """Evaluate model performance on the test dataset

        Args:
            state_dict_path (str): file containing the weights to be evaluated
        """
        eval = TLMEvaluator(
            self.test_loader, state_dict_path, self.device, self.threshold
        )
        self.mean_ious.append(eval.compute_mean_iou())
        print(f"Mean AP on test dataset: {self.mean_ious[-1]}")

    @torch.no_grad()
    def histogram(self, state_dict_path: str, hist_file: str) -> None:
        """Create histogram of IoU scores for each image in the test dataset

        Args:
            state_dict_path (str): file containing model weights to be evaluated
            hist_file (str): file to save the histogram to
        """
        eval = TLMEvaluator(
            self.test_loader, state_dict_path, self.device, self.threshold
        )
        mean = eval.make_histogram(hist_file)
        self.mean_ious.append(mean)
        print(f"Mean AP on test dataset: {self.mean_ious[-1]}")

    def create_optimiser(self, learning_rate) -> optim.Adam:
        params = self.model.parameters()
        return optim.Adam(params, lr=learning_rate)

    def create_lr_scheduler(self) -> optim.lr_scheduler.OneCycleLR:
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=0.01,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader),
        )
        return scheduler

    def create_warmup_lr_scheduler(
        self, warmup_factor: float
    ) -> optim.lr_scheduler.LambdaLR:
        # define custom function to build the scheduler with - here the first epoch will use a linear schedule
        def f(x):
            iters = len(self.train_loader)
            if x >= iters:
                return 1
            alpha = float(x) / iters
            return warmup_factor * (1 - alpha) + alpha

        return optim.lr_scheduler.LambdaLR(self.optimiser, f)

    def train_one_epoch(self, epoch_no: int) -> None:
        """Completes one epoch of training, including evaluating test losses

        Args:
            epoch_no (int): number of the current epoch being trained
        """
        self.model.train()

        # run warm-up linear scheduler for the first epoch, otherwise use the main scheduler
        if epoch_no == 0:
            lr_scheduler = self.create_warmup_lr_scheduler(0.001)
        else:
            lr_scheduler = self.lr_scheduler

        # could be generalised with the DataParallel functionality to train on multiple GPUs
        for batch_imgs, batch_targets in self.train_loader:
            batch_imgs = [batch_img.to(self.device) for batch_img in batch_imgs]
            batch_targets = [
                {key: val.to(self.device) for key, val in t.items()}
                for t in batch_targets
            ]

            loss_dict: dict[Any, Tensor] = self.model(batch_imgs, batch_targets)
            losses = torch.as_tensor(sum(loss for loss in loss_dict.values()))
            self.train_losses.append(losses.item())

            self.optimiser.zero_grad()
            losses.backward()
            self.optimiser.step()
            lr_scheduler.step()

        if epoch_no % self.save_freq == 0:
            filename = f"{self.folder}/{epoch_no}.torch"
            torch.save(self.model.state_dict(), filename)
            self.last_save_no = epoch_no
            print(f"State saved at epoch {epoch_no}")

        self.avg_train_losses.append(np.average(self.train_losses))

    @torch.no_grad()
    def calc_test_losses(self) -> None:
        """Calculate losses on the test dataset"""
        for batch_imgs, batch_targets in self.test_loader:
            batch_imgs = [batch_img.to(self.device) for batch_img in batch_imgs]
            batch_targets = [
                {key: val.to(self.device) for key, val in t.items()}
                for t in batch_targets
            ]

            loss_dict: dict[Any, Tensor] = self.model(batch_imgs, batch_targets)
            losses = torch.as_tensor(sum(loss for loss in loss_dict.values()))
            self.test_losses.append(losses.item())
        self.avg_test_losses.append(np.average(self.test_losses))

    def early_stopping(self, min_delta: float) -> bool:
        """Check if the change in loss is small enough that the model has converged

        Args:
            min_delta (float): minimum change in loss between epochs to consider the model converged

        Returns:
            bool: whether the model has converged or not
        """
        if (
            self.avg_test_losses[-1] > (self.avg_test_losses[-2] - min_delta)
            and self.avg_test_losses[-1] < self.avg_test_losses[-2]
        ):
            return True
        else:
            return False

    def output_loss_fig(self, file_id: str) -> None:
        """Save out a figure showing training and validation loss versus
        epoch number.

        Args:
            model_out_path (Path): path to the model output by the trainer.
        """

        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            range(1, len(self.avg_train_losses) + 1),
            self.avg_train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(self.avg_test_losses) + 1),
            self.avg_test_losses,
            label="Validation Loss",
        )

        minposs = (
            self.avg_test_losses.index(min(self.avg_test_losses)) + 1
        )  # find position of lowest validation loss
        plt.axvline(
            minposs, linestyle="--", color="r", label="Early Stopping Checkpoint"
        )

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xlim(0, len(self.avg_train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig_out_pth = f"{self.folder}/{file_id}_loss_plot.png"
        print(f"Saving figure of training/validation losses to {fig_out_pth}")
        fig.savefig(fig_out_pth, bbox_inches="tight")
        # Output a list of training stats
        epoch_lst = range(len(self.avg_train_losses))
        rows = zip(
            epoch_lst,
            self.avg_train_losses,
            self.avg_test_losses,
            self.mean_ious,
        )
        with open(f"{self.folder}/{file_id}_train_stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Epoch", "Train Loss", "Valid Loss", "Eval Score"))
            for row in rows:
                writer.writerow(row)
