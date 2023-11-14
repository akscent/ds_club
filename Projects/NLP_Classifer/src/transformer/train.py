class Trainer:
    def __init__(self, config: Dict, class_weights=None):
        self.config = config
        self.device = config["device"]
        self.n_epochs = config["n_epochs"]
        self.optimizer = None
        self.opt_fn = lambda model: Adamax(model.parameters(), config["lr"])
        self.model = None
        self.history = None
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.loss_fn = CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = CrossEntropyLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", True)

    def save_history(self, path: str):
        history = {
            "train_loss": self.history["train_loss"],
            "val_loss": self.history["val_loss"],
            "val_acc": self.history["val_acc"],
        }
        val_acc = sum(self.history["val_acc"]) / len(self.history["val_acc"])
        print("All ACCURACY = ", val_acc)
        with open(path, "w") as file:
            json.dump(history, file)

    def load_history(self, path: str):
        with open(path, "r") as file:
            history = json.load(file)
        self.history = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
        }

    def fit(self, model, train_dataloader, val_dataloader):
        self.model = model.to(self.device)
        self.optimizer = self.opt_fn(model)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            train_info = self.train_epoch(train_dataloader)
            val_info = self.val_epoch(val_dataloader)
            self.history["train_loss"].extend(train_info["loss"])
            self.history["val_loss"].extend([val_info["loss"]])
            self.history["val_acc"].extend([val_info["acc"]])

            if val_info["loss"] < best_val_loss:
                best_val_loss = val_info["loss"]
                self.save_model_weights("best_model_weights.ckpt")

            self.save_history("history.json")

        return self.model.eval()

    def save_model_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []
        total_loss = 0
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        for batch in train_dataloader:
            ids = batch["ids"].to(self.device, dtype=torch.long)
            mask = batch["mask"].to(self.device, dtype=torch.long)
            targets = batch["targets"].to(self.device, dtype=torch.long)

            outputs = self.model(ids, mask)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            if self.verbose:
                train_dataloader.set_description(f"Loss={loss_val:.3}")
            losses.append(loss_val)
        avg_loss = total_loss / len(train_dataloader)
        print("AVG LOSS = ", avg_loss)
        return {"loss": losses}

    def val_epoch(self, val_dataloader):
        self.model.eval()
        all_logits = []
        all_labels = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)
        with torch.no_grad():
            for batch in val_dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long)
                targets = batch["targets"].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask)
                all_logits.append(outputs)
                all_labels.append(targets)
        all_labels = torch.cat(all_labels).to(self.device)
        all_logits = torch.cat(all_logits).to(self.device)
        loss = self.loss_fn(all_logits, all_labels).item()
        acc = (all_logits.argmax(1) == all_labels).float().mean().item()
        print("ACCURACY for EPOCH = ", acc)
        if self.verbose:
            val_dataloader.set_description(f"Loss={loss:.3}; Acc:{acc:.3}")
        return {"acc": acc, "loss": loss}

    def predict(self, test_dataloader):
        if not self.model:
            raise RuntimeError("You should train the model first")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask)
                preds = torch.exp(outputs)
                predictions.extend(preds.tolist())
        return asarray(predictions)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {
            "config": self.model.config,
            "trainer_config": self.config,
            "model_name": self.model.model_name,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)

    def plot_history(self):
        import matplotlib.pyplot as plt

        if self.history is None:
            raise RuntimeError("History is not available. Train the model first.")

        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]
        val_acc = self.history["val_acc"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_acc, "g", label="Validation accuracy")
        plt.title("Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["config", "trainer_config", "model_state_dict"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = ModelForClassification(ckpt["model_name"], ckpt["config"])
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.model.to(new_trainer.device)
        return new_trainer
