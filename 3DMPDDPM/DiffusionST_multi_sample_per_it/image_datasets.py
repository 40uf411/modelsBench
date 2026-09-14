from PIL import Image
import blobfile as bf
# from mpi4py import MPI  # Commented out to remove MPI dependency
import numpy as np
from torch.utils.data import DataLoader, Dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        arr = np.load(path, allow_pickle=True)
        arr = arr.astype(np.float32)
        # Given that the data is 3D, if it the channel dimension is not set then add it (i.e. (50, 150, 150) -> (1, 50, 150, 150)))
        if len(arr.shape) == 3:
            arr = np.expand_dims(arr, 0)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict


def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        all_files,
        classes=classes,
        # shard=MPI.COMM_WORLD.Get_rank(),  # Commented out to remove MPI dependency
        # num_shards=MPI.COMM_WORLD.Get_size(),  # Commented out to remove MPI dependency
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
