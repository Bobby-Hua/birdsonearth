from torch.utils.data import Dataset
import random


class MelDataset(Dataset):

    def __init__(self, mel_dict, sample_len):
        self._sample_len = sample_len
        self.labels = sorted(list(mel_dict.keys()))
        self._insts = []
        for k, v in mel_dict.items():
            for inst in v:
                self._insts.append(
                    (inst, self.labels.index(k))
                )

    def _random_crop(self, mel):
        f_start = random.randint(0, mel.shape[0] - self._sample_len)
        chunk = mel[:, f_start: f_start + self._sample_len]
        return chunk

    def __getitem__(self, idx):
        mel, label = self._insts[idx]
        chunk = self._random_crop(mel)
        return chunk, label

    def __len__(self):
        return len(self._insts)