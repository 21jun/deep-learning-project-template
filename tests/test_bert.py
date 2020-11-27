from pytorch_lightning import Trainer, seed_everything
from project.lit_text_classifier import LitBertClassifier
from project.lit_text_classifier import LitTweetDataModule


def test_text_classifier():
    seed_everything(1234)

    model = LitBertClassifier(n_classes=2)
    tweet = LitTweetDataModule(data_dir="data/TWEET/train.csv", batch_size=8, max_len=100)

    trainer = Trainer(limit_train_batches=0.01, limit_val_batches=0.01, limit_test_batches=0.01, max_epochs=1)
    trainer.fit(model, tweet)

    results = trainer.test(datamodule=tweet)

    assert results[0]['test_acc'] > 0.7


if __name__ == "__main__":
    test_text_classifier()
