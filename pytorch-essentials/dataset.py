from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

# __len__()       → how many samples?
# __getitem__()   → give me sample i

# -----------------------------------------------------------------------------

# It takes your Dataset and gives you batches.
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


for X_batch, y_batch in loader:
    ...

