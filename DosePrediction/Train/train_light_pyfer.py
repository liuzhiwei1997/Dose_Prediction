import os
import gc
from pathlib import Path
from typing import Optional

import DosePrediction.Train.config as config
import SimpleITK as sitk
from monai.data import DataLoader, list_data_collate

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from DosePrediction.Models.Networks.dose_pyfer import *
from DosePrediction.DataLoader.dataloader_OpenKBP_monai import get_dataset
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Train.loss import GenLoss
from DosePrediction.utils.runtime import (
    get_bitsandbytes_module,
    get_lightning_accelerator,
    resolve_optional_checkpoint,
    resolve_output_dir,
    use_bitsandbytes,
    use_pin_memory,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = torch.cuda.is_available()


def use_cached_dataset():
    return os.environ.get("DOSE_PREDICTION_USE_CACHE", "1") == "1"


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.val_data = None

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        cache_enabled = use_cached_dataset()
        self.train_data = get_dataset(path=os.path.join(config.MAIN_PATH, config.TRAIN_DIR), state='train',
                                      size=config.TRAIN_SIZE, cache=cache_enabled, crop_flag=False)

        self.val_data = get_dataset(path=os.path.join(config.MAIN_PATH, config.VAL_DIR), state='val',
                                    size=config.VAL_SIZE, cache=cache_enabled)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=use_pin_memory())

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=use_pin_memory())


class TestOpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign val datasets for use in dataloaders
        cache_enabled = use_cached_dataset()
        self.test_data = get_dataset(path=os.path.join(config.MAIN_PATH, config.TEST_DIR), state='test',
                                     size=config.VAL_SIZE, cache=cache_enabled)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=use_pin_memory())


class Pyfer(pl.LightningModule):
    def __init__(
            self,
            config_param,
            freeze=True
    ):
        super().__init__()
        self.config_param = config_param
        self.freeze = freeze
        self.save_hyperparameters()

        # OAR + PTV + CT => dose

        self.model_, inside = create_pretrained_unet(
            in_ch=9, out_ch=1,
            list_ch_A=[-1, 16, 32, 64, 128, 256],
            ckpt_file=resolve_optional_checkpoint(
                'DOSE_PREDICTION_PRETRAINED_C3D',
                'PretrainedModels/DosePrediction/C3D_bs4_iter80000.pkl',
                'PretrainedModels/C3D_bs4_iter80000.pkl',
            ),
            feature_size=16,
            img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
            num_layers=8,  # 4, 8, 12
            num_heads=6,  # 3, 6, 12
            act=config_param["act"],
            mode_multi_dec=True,
            multiS_conv=config_param["multiS_conv"], )

        if freeze:
            for n, param in self.model_.named_parameters():
                if 'net_A' in n or 'conv_out_A' in n:
                    param.requires_grad = False

        self.lr = config_param["lr"]
        self.weight_decay = config_param["weight_decay"]
        self.hotspot_weight = config_param.get("hotspot_weight", 0.75)
        self.hotspot_quantile = config_param.get("hotspot_quantile", 0.98)
        self.coldspot_weight = config_param.get("coldspot_weight", 0.35)
        self.coldspot_quantile = config_param.get("coldspot_quantile", 0.10)

        self.loss_function = GenLoss(im_size=config.IMAGE_SIZE)
        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None
        self.metric_values = []

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss = []

        self.max_epochs = 1300
        self.check_val = 5
        self.warmup_epochs = 1

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE

        self.sw_batch_size = config.SW_BATCH_SIZE

        self.list_DVH_dif = []
        self.list_dose_metric = []
        self.dict_DVH_dif = {}
        self.ivs_values = []

    def forward(self, x):
        return self.model_(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch['Input'].float()
        target = batch['GT']

        # train
        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target, casecade=True, freez=self.freeze,
                                  delta1=self.config_param['delta1'], delta2=self.config_param['delta2'],
                                  hotspot_weight=self.hotspot_weight, hotspot_quantile=self.hotspot_quantile,
                                  coldspot_weight=self.coldspot_weight, coldspot_quantile=self.coldspot_quantile)

        if self.moving_train_loss is None:
            self.moving_train_loss = loss.item()
        else:
            self.moving_train_loss = \
                (1 - self.eps_train_loss) * self.moving_train_loss \
                + self.eps_train_loss * loss.item()

        tensorboard_logs = {"train_loss": loss.item()}

        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if train_mean_loss < self.best_average_train_loss:
            self.best_average_train_loss = train_mean_loss
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())
        self.logger.log_metrics({"train_mean_loss": train_mean_loss}, self.current_epoch + 1)
        torch.cuda.empty_cache()

    def validation_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        prediction = self.forward(input_)

        torch.cuda.empty_cache()
        loss = self.loss_function(prediction[1][0], target, mode='val', casecade=True, freez=self.freeze)

        prediction_b = np.array(prediction[1][0].cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 70. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))

        return {"val_loss": loss, "val_metric": dose_score}

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_dose_score = - np.stack([x["val_metric"] for x in outputs]).mean()
        if mean_dose_score > self.best_average_val_index:
            self.best_average_val_index = mean_dose_score
        self.metric_values.append(mean_dose_score)

        self.log("mean_dose_score", mean_dose_score, logger=False)
        self.log("val_loss", avg_loss, logger=False)
        self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)

        tensorboard_logs = {"val_metric": mean_dose_score}
        torch.cuda.empty_cache()

        return {"log": tensorboard_logs}

    def configure_optimizers(self):
        bnb = get_bitsandbytes_module()
        if use_bitsandbytes() and bnb is not None:
            optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=self.lr,
                                           weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        return optimizer

    def test_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]

        gt_dose = target[:, :1, :, :, :].cpu()
        possible_dose_mask = target[:, 1:, :, :, :].cpu()

        prediction = self.forward(input_)
        prediction = prediction[1][0].cpu()

        invalid_mask = torch.logical_or(possible_dose_mask < 1, prediction < 0)
        prediction[invalid_mask] = 0

        prediction = 70. * prediction

        dose_dif, dvh_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif,
            ivs_values=self.ivs_values)
        self.list_DVH_dif.append(dvh_dif)

        torch.cuda.empty_cache()
        save_prediction_nifti = os.environ.get("DOSE_PREDICTION_SAVE_TEST_NIFTI", "1") == "1"
        save_test_dvh = os.environ.get("DOSE_PREDICTION_SAVE_TEST_DVH", "0") == "1"
        save_test_jpg = os.environ.get("DOSE_PREDICTION_SAVE_TEST_JPG", "0") == "1"

        name_p = Path(batch_data['file_path'][0]).parent.name

        if save_prediction_nifti:
            prediction_dir = Path(resolve_output_dir('DosePrediction', 'predictions', name_p))
            prediction_nii = sitk.GetImageFromArray(prediction[0, 0].numpy().astype(np.float32))
            reference_nii = sitk.ReadImage(batch_data['file_path'][0])
            prediction_nii.CopyInformation(reference_nii)
            sitk.WriteImage(prediction_nii, str(prediction_dir / 'dose.nii.gz'))

        if batch_idx < 100:
            ckp_re_dir = resolve_output_dir('DosePrediction', 'test_outputs', 'ours_model')

            if save_test_dvh:
                plot_DVH(prediction, batch_data, path=os.path.join(ckp_re_dir, 'dvh_{}.png'.format(batch_idx)))

            if save_test_jpg:
                gt_dose[possible_dose_mask < 1] = 0

                predicted_img = torch.permute(prediction[0].cpu(), (1, 0, 2, 3))
                gt_img = torch.permute(gt_dose[0], (1, 0, 2, 3))
                for i in range(len(predicted_img)):
                    predicted_i = predicted_img[i][0].numpy()
                    gt_i = 70. * gt_img[i][0].numpy()
                    error = np.abs(gt_i - predicted_i)

                    # Create a figure and axis object using Matplotlib
                    fig, axs = plt.subplots(3, 1, figsize=(4, 10))
                    plt.subplots_adjust(wspace=0, hspace=0)

                    # Display the ground truth array
                    axs[0].imshow(gt_i, cmap='jet')
                    axs[0].axis('off')

                    # Display the prediction array
                    axs[1].imshow(predicted_i, cmap='jet')
                    axs[1].axis('off')

                    # Display the error map using a heatmap
                    axs[2].imshow(error, cmap='jet')
                    axs[2].axis('off')

                    save_dir = Path(ckp_re_dir) / f"{name_p}_{batch_idx}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    fig.savefig(save_dir / f"{i}.jpg", bbox_inches="tight")

                    torch.cuda.empty_cache()
                    gc.collect()

        self.list_dose_metric.append(dose_dif)
        return {"dose_dif": dose_dif}

    def test_epoch_end(self, outputs):

        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        std_dose_metric = np.stack([x["dose_dif"] for x in outputs]).std()
        mean_dvh_metric = np.mean(self.list_DVH_dif)

        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)

        self.ivs_values = np.array(self.ivs_values)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("std_dose_metric", std_dose_metric)
        return self.dict_DVH_dif


def build_logger(run_id=None, run_name=None):
    logger_type = os.environ.get("DOSE_PREDICTION_LOGGER", "csv").lower()
    if logger_type == "mlflow":
        tracking_uri = os.environ.get("DOSE_PREDICTION_MLFLOW_TRACKING_URI", "file:" + resolve_output_dir("mlruns"))
        return MLFlowLogger(
            experiment_name=os.environ.get("DOSE_PREDICTION_EXPERIMENT", "dose_prediction"),
            tracking_uri=tracking_uri,
            run_id=run_id,
            run_name=run_name,
        )

    return CSVLogger(save_dir=resolve_output_dir("logs"), name="dose_prediction")


def main(freeze=True, delta1=10, delta2=8, run_id=None, run_name=None, ckpt_path=None,
         max_epochs=None, fast_dev_run=False):
    openkbp_ds = OpenKBPDataModule()
    config_param = {
        "act": 'mish',
        "multiS_conv": True,
        "lr": 0.0006130697604327541,
        'weight_decay': 0.00016303111017674179,
        'delta1': delta1,
        'delta2': delta2,
    }
    net = Pyfer(
        config_param,
        freeze=freeze
    )

    ckpt_dir = ckpt_path or config.CHECKPOINT_MODEL_DIR_FINAL
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
    )

    accelerator, devices = get_lightning_accelerator()
    logger = build_logger(run_id=run_id, run_name=run_name)

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=max_epochs or net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=ckpt_dir,
        fast_dev_run=fast_dev_run,
    )

    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    resume_ckpt = last_ckpt if (run_name is None and os.path.exists(last_ckpt)) else None
    trainer.fit(net, datamodule=openkbp_ds, ckpt_path=resume_ckpt)

    return net


if __name__ == '__main__':
    main()
