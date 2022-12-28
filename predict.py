from deploy.model import ConformerRNNModel
from mpvn.configs import DictConfig
from mpvn.vocabs.grad import GradVocabulary
import json

configs = DictConfig()
vocab = GradVocabulary(f"Data/token.txt")
phone_map = json.load(open("Data/phone_map.json"))

model = ConformerRNNModel.load_from_checkpoint(
    'Checkpoint/finetuned-epoch=24-valid_loss=0.14-valid_per=0.47-valid_acc=0.93-valid_f1=0.68.ckpt',
    configs=configs,
    num_classes=len(vocab),
    vocab=vocab,
    phone_map=phone_map
)
model.eval()

audio_path = 'Data/label/Audio/2022-12-12-AFRO-trym/2022-12-12-AFRO-trym_42.wav'
transcript = 'quy nhơn'

print(model.predict(audio_path, transcript))