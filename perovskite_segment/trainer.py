import os
import torch
import torch.optim as optim

try:
    import wandb
    wandb.init(
    project="perovskite-segment",
    name=None,
)

    USE_WANDB = True

except ImportError:

    USE_WANDB = False


class Trainer:

    def __init__(
        self,
        cfg,
        train_loader,
        model,
        evaluator,
        criterion,
        device,
    ):

        self.cfg = cfg

        self.train_loader = train_loader

        self.model = model

        self.evaluator = evaluator

        self.criterion = criterion

        self.device = device

        # hyperparameters

        self.epochs = cfg.epochs

        self.lr = cfg.lr

        self.weight_decay = cfg.weight_decay

        self.log_interval = cfg.log_interval

        self.eval_interval = cfg.eval_interval

        self.grad_clip = getattr(
            cfg,
            "grad_clip",
            None,
        )

        self.checkpoint_dir = (
            cfg.checkpoint_dir
        )

        # optimization

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        )


        self.best_dice = 0

    def save_checkpoint(
        self,
        epoch,
        tag="best",
    ):

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        path = os.path.join(
            self.checkpoint_dir,
            f"unet_{tag}.pt",
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state":
                    self.model.state_dict(),

                "optimizer_state":
                    self.optimizer.state_dict(),

                "best_dice":
                    self.best_dice,

                "cfg":
                    self.cfg,
            },
            path,
        )

        print(
            f"Saved checkpoint:"
            f" {path}"
        )


    def train_step(
        self,
        images,
        masks,
    ):

        images = images.to(
            self.device
        )

        masks = masks.to(
            self.device
        )

        self.optimizer.zero_grad()

        outputs = self.model(
            images
        )

        outputs = self.model(images)

        if self.cfg.model.name == "deeplabv3_resnet50":

            prediction = outputs["out"]

            loss = self.criterion(
                prediction,
                masks
            )
        else:

            if isinstance(outputs, (tuple, list)):
                # U²-Net
                loss = torch.stack([
                    self.criterion(pred, masks)
                    for pred in outputs
                ]).mean()
                prediction = outputs[0]  # final output for metrics
            else:
                # U-Net
                prediction = outputs
                loss = self.criterion(
                    prediction,
                    masks,
                )

        loss.backward()
        if self.grad_clip:

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        epoch,
    ):

        self.model.train()

        running_loss = 0

        for step, (
            images,
            masks,
        ) in enumerate(
            self.train_loader
        ):

            loss = self.train_step(
                images,
                masks,
            )

            running_loss += loss

            if (
                step
                %
                self.log_interval
                ==
                0
            ):

                print(
                    f"Epoch "
                    f"[{epoch}/"
                    f"{self.epochs}] "
                    f"Step "
                    f"[{step}/"
                    f"{len(self.train_loader)}] "
                    f"Loss: "
                    f"{loss:.4f}"
                )

        avg_loss = (
            running_loss
            /
            len(
                self.train_loader
            )
        )

        return avg_loss


    def train(self):

        print(
            "\nStarting training\n"
        )

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            # training

            train_loss = (
                self.train_epoch(
                    epoch
                )
            )

            print(
                f"\nEpoch "
                f"{epoch}"
                f" train loss:"
                f" {train_loss:.4f}"
            )

            # validation

            if (
                epoch
                %
                self.eval_interval
                ==
                0
            ):

                metrics = (
                    self.evaluator
                    .eval()
                )

                # scheduler

                self.scheduler.step(
                    metrics[
                        "val_dice"
                    ]
                )

                # logging

                if USE_WANDB:
                    wandb.log(
                        {
                            "epoch":
                                epoch,

                            "train_loss":
                                train_loss,

                            **metrics,
                        }
                    )

                # checkpoint

                if (
                    metrics[
                        "val_dice"
                    ]
                    >
                    self.best_dice
                ):

                    self.best_dice = (
                        metrics[
                            "val_dice"
                        ]
                    )

                    self.save_checkpoint(
                        epoch,
                        "best",
                    )

        # final checkpoint

        self.save_checkpoint(
            self.epochs,
            "final",
        )

        print(
            "\nTraining finished"
        )