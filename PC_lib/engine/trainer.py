from PC_lib.model import PC_BASELINE
from PC_lib.data import PCDataset
from PC_lib.utils.utils import AvgRecorder, duration_in_hours
import logging
import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from time import time
import os
import h5py

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class PCTrainer:
    def __init__(self, args, max_K, category_number):
        self.args = args
        self.log = logging.getLogger("Network")
        # data_path is a dictionary {'train', 'test'}
        if args.device == "cuda:0" and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.device = device
        self.log.info(f"Using device {self.device}")

        self.max_K = max_K
        self.category_number = category_number

        self.max_epochs = args.max_epochs
        self.model = self.build_model()
        self.model.to(device)
        self.log.info(f"Below is the network structure:\n {self.model}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, betas=(0.9, 0.99)
        )
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)

        self.data_path = args.data_path
        self.writer = None

        self.train_loader = None
        self.test_loader = None
        self.init_data_loader(args.test)
        self.test_result = None

    def build_model(self):
        model = PC_BASELINE(self.max_K, self.category_number)
        return model

    def init_data_loader(self, eval_only):
        if not eval_only:
            self.train_loader = torch.utils.data.DataLoader(
                PCDataset(
                    self.data_path["train"], num_points=self.args.num_points, max_K=self.max_K
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
            )

            self.log.info(f'Num {len(self.train_loader)} batches in train loader')

        self.test_loader = torch.utils.data.DataLoader(
            PCDataset(
                self.data_path["test"], num_points=self.args.num_points, max_K=self.max_K
            ),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        self.log.info(f'Num {len(self.test_loader)} batches in test loader')

    def train_epoch(self, epoch):
        self.log.info(f'>>>>>>>>>>>>>>>> Train Epoch {epoch} >>>>>>>>>>>>>>>>')

        self.model.train()

        iter_time = AvgRecorder()
        io_time = AvgRecorder()
        to_gpu_time = AvgRecorder()
        network_time = AvgRecorder()
        start_time = time()
        end_time = time()
        remain_time = ''

        epoch_loss = {
            'total_loss': AvgRecorder()
        }

        # if self.train_loader.sampler is not None:
        #     self.train_loader.sampler.set_epoch(epoch)
        for i, (camcs_per_point, gt_dict, id) in enumerate(self.train_loader):
            io_time.update(time() - end_time)
            # Move the tensors to the device
            s_time = time()
            camcs_per_point = camcs_per_point.to(self.device)
            gt = {}
            for k, v in gt_dict.items():
                gt[k] = v.to(self.device)
            to_gpu_time.update(time() - s_time)

            # Get the loss
            s_time = time()
            pred = self.model(camcs_per_point)
            loss_dict = self.model.losses(pred, camcs_per_point, gt)
            network_time.update(time() - s_time)

            loss = torch.tensor(0.0, device=self.device)
            loss_weight = self.args.loss_weight
            # use different loss weight to calculate the final loss
            for k, v in loss_dict.items():
                if k not in loss_weight:
                    raise ValueError(f"No loss weight for {k}")
                loss += loss_weight[k] * v

            # Used to calculate the avg loss
            for k, v in loss_dict.items():
                if k not in epoch_loss.keys():
                    epoch_loss[k] = AvgRecorder()
                epoch_loss[k].update(v)
            epoch_loss['total_loss'].update(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # time and print
            current_iter = epoch * len(self.train_loader) + i + 1
            max_iter = (self.max_epochs + 1) * len(self.train_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(time() - end_time)
            end_time = time()

            remain_time = remain_iter * iter_time.avg
            remain_time = duration_in_hours(remain_time)

        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
        # self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)
        # self.scheduler.step()
        # Add the loss values into the tensorboard
        for k, v in epoch_loss.items():
            if k == "total_loss":
                self.writer.add_scalar(f"{k}", epoch_loss[k].avg, epoch)
            else:
                self.writer.add_scalar(f"loss/{k}", epoch_loss[k].avg, epoch)
        if epoch % self.args.log_frequency == 0:
            loss_log = ''
            for k, v in epoch_loss.items():
                loss_log += '{}: {:.5f}  '.format(k, v.avg)
            self.log.info(
                'Epoch: {}/{} Loss: {} io_time: {:.2f}({:.4f}) to_gpu_time: {:.2f}({:.4f}) network_time: {:.2f}({:.4f}) \
                duration: {:.2f} remain_time: {}'
                    .format(epoch, self.max_epochs, loss_log, io_time.sum, io_time.avg, to_gpu_time.sum,
                            to_gpu_time.avg, network_time.sum, network_time.avg, time() - start_time, remain_time))

    def eval_epoch(self, epoch, save_results=False):
        self.log.info(f'>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        val_error = {
            'total_loss': AvgRecorder()
        }
        if save_results:
            existDir(self.args.output_dir)
            inference_path = f"{self.args.output_dir}/inference_result.h5"
            self.test_result = h5py.File(inference_path, "w")

        # test the model on the val set and write the results into tensorboard
        self.model.eval()
        with torch.no_grad():
            start_time = time()
            for i, (camcs_per_point, gt_dict, id) in enumerate(self.test_loader):
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)

                pred = self.model(camcs_per_point)
                if save_results:
                    self.save_results(pred, camcs_per_point, gt, id)
                loss_dict = self.model.losses(pred, camcs_per_point, gt)
                loss_weight = self.args.loss_weight
                loss = torch.tensor(0.0, device=self.device)
                # use different loss weight to calculate the final loss
                for k, v in loss_dict.items():
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v

                # Used to calculate the avg loss
                for k, v in loss_dict.items():
                    if k not in val_error.keys():
                        val_error[k] = AvgRecorder()
                    val_error[k].update(v)
                val_error['total_loss'].update(loss)
        # write the val_error into the tensorboard
        if self.writer is not None:
            for k, v in val_error.items():
                self.writer.add_scalar(f"val_error/{k}", val_error[k].avg, epoch)

        loss_log = ''
        for k, v in val_error.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)

        self.log.info(
            'Eval Epoch: {}/{} Loss: {} duration: {:.2f}'
                .format(epoch, self.max_epochs, loss_log, time() - start_time))
        if save_results:
            self.test_result.close()
        return val_error

    def train(self, start_epoch=0):
        self.model.train()
        self.writer = SummaryWriter(self.args.output_dir)

        existDir(self.args.output_dir)

        best_model = None
        best_result = np.inf
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.args.save_frequency == 0 or epoch == self.max_epochs:
                # Save the model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    f"{self.args.output_dir}/model_{epoch}.pth"
                )

                val_error = self.eval_epoch(epoch)

                if best_model is None or val_error["total_loss"].avg < best_result:
                    best_model = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }
                    best_result = val_error["total_loss"].avg
                    torch.save(
                        best_model,
                        f"{self.args.output_dir}/best_model.pth"
                    )
        self.writer.close()

    def test(self, inference_model=None):
        # Load the model
        self.log.info(f"Load model from {inference_model}")
        checkpoint = torch.load(inference_model, map_location=self.device)
        epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.eval_epoch(epoch, save_results=True)


    def save_results(self, pred, camcs_per_point, gt, id):
        # Save the results and gt into hdf5 for further optimization
        batch_size = pred["category_per_point"].shape[0]
        for b in range(batch_size):
            group = self.test_result.create_group(f"{id[b]}")
            group.create_dataset(
                "camcs_per_point",
                data=camcs_per_point[b].detach().cpu().numpy(),
                compression="gzip",
            )

            # save prediction results
            # Save the predicted part category
            pred_category_per_point = np.argmax(pred['category_per_point'][b].detach().cpu().numpy(), axis=1)
            group.create_dataset('pred_category_per_point', data=pred_category_per_point, compression="gzip")
            # Save the predictd instance id
            pred_instance_per_point = np.argmax(pred['instance_per_point'][b].detach().cpu().numpy(), axis=1)
            group.create_dataset('pred_instance_per_point', data=pred_instance_per_point, compression="gzip")
            # Save the predicted motion type
            pred_mtype_per_point = np.argmax(pred['mtype_per_point'][b].detach().cpu().numpy(), axis=1)
            group.create_dataset('pred_mtype_per_point', data=pred_mtype_per_point, compression="gzip")
            # Save the motion axis and motion origin
            group.create_dataset('pred_maxis_per_point', data=pred['maxis_per_point'][b].detach().cpu().numpy(), compression="gzip")
            group.create_dataset('pred_morigin_per_point', data=pred['morigin_per_point'][b].detach().cpu().numpy(), compression="gzip")

            # Save the gt
            for k, v in gt.items():
                if k == "num_instances":
                    group.create_dataset(
                        f"gt_{k}", data=[gt[k][b].detach().cpu().numpy()], compression="gzip"
                    )
                else:
                    group.create_dataset(
                        f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
                    )

    def resume_train(self, model_path):
        # Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        epoch = checkpoint["epoch"]
        self.log.info(f"Continue training with model from {model_path} at epoch {epoch}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)

        self.train(epoch)
