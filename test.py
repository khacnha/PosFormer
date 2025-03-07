from pytorch_lightning import Trainer
from Pos_Former.datamodule import CROHMEDatamodule
from Pos_Former.lit_posformer import LitPosFormer


checkpoint_path = "lightning_logs/version_0/checkpoints/best.ckpt"

datamodule = CROHMEDatamodule(test_year='2014', eval_batch_size=1)

model = LitPosFormer.load_from_checkpoint(checkpoint_path)

trainer = Trainer(gpus=0)

trainer.test(model, datamodule=datamodule)
