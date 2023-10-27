from collections import OrderedDict, namedtuple

import numpy as np
import torch

# from neural_sampling_code.elements.loaders import ImageNamedTupleWrappedDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from nnsysident.datasets import mouse_loaders


class ImageNamedTupleWrappedDataset(Dataset):
    """
    Wraps a dataset of images and targets into a dataset of named tuples.
    The named tuple contains the images as 'inputs' and the 'targets' as targets.
    """

    def __init__(self, named_tuple_class, dataset, image_shape=(1, 41, 41)):
        self.named_tuple_class = named_tuple_class
        self.dataset = dataset
        self.image_shape = image_shape
        # self.neurons = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        inputs = inputs.view(self.image_shape)
        return self.named_tuple_class(inputs, targets)


def get_common_unit_ids(basepath, sessions):
    multi_id_suffix = "meta/neurons/multi_match_id.npy"  # multi unit ids are ids that are shared across sessions
    unit_id_suffix = (
        "meta/neurons/unit_ids.npy"  # unit ids are ids that are unique to a session
    )
    # load multi and unit ids for each session
    multi_id_paths = [basepath + session + multi_id_suffix for session in sessions]
    unit_id_paths = [basepath + session + unit_id_suffix for session in sessions]
    multi_ids = [np.load(filename) for filename in multi_id_paths]
    unit_ids = [np.load(filename) for filename in unit_id_paths]
    # get multi-ids that are shared across sessions
    common_multi_ids = list(set.intersection(*map(set, multi_ids)))
    # remove -1 from common multi-ids (these are units that are not shared across sessions)
    common_multi_ids = [
        common_multi_id for common_multi_id in common_multi_ids if common_multi_id != -1
    ]
    # now find the indices of the common multi-ids in each session
    multi_id_indices = [
        [np.where(multi_id == cmi)[0][0] for cmi in common_multi_ids]
        for multi_id in multi_ids
    ]
    return [
        unit_id[multi_id_idx]
        for unit_id, multi_id_idx in zip(unit_ids, multi_id_indices)
    ]


def get_matched_mouse_dataloaders(
    basepath="/data/mouse/toliaslab/static/",
    sessions=[
        "static22564-3-12-preproc0/",
        "static22564-3-8-preproc0/",
        "static22564-2-13-preproc0/",
        "static22564-2-12-preproc0/",
    ],
    normalize_per_session=True,
    normalize_across_sessions=False,
    batch_size=128,
    cuda=False,
    seed=42,
    combine_into_one_dataloader=False,
):
    """
    Fetches combined dataloaders for matched sessions of mouse data.
    """
    print("Fetching matched mouse dataloaders...")
    # create full directory paths for each session
    dirnames = [basepath + session for session in sessions]
    # get common unit ids across sessions (these are the units that are shared across sessions)
    common_unit_ids = get_common_unit_ids(basepath, sessions)
    # use these common unit ids to fetch the dataloaders
    dataset_config = {
        "paths": dirnames,
        "normalize": normalize_per_session,
        "neuron_ids": common_unit_ids,
        "batch_size": batch_size,
        "cuda": cuda,
        "seed": seed,
    }
    dataloaders = mouse_loaders.static_loaders(**dataset_config)
    if not combine_into_one_dataloader:
        final_dataloaders = dataloaders
    else:
        trainloader = dataloaders["train"]
        validationloader = dataloaders["validation"]
        testloader = dataloaders["test"]
        train_images, train_responses = [], []
        validation_images, validation_responses = [], []
        test_images, test_responses = [], []
        print("Concatenating images and responses from matched sessions...")
        for session in trainloader.keys():
            print("Session: ", session)
            print("Train")
            for images, responses in tqdm(trainloader[session]):
                train_images.append(images)
                train_responses.append(responses)
            print("Validation")
            for images, responses in tqdm(validationloader[session]):
                validation_images.append(images)
                validation_responses.append(responses)
            print("Test")
            for images, responses in tqdm(testloader[session]):
                test_images.append(images)
                test_responses.append(responses)

        train_images = torch.cat(train_images, dim=0)
        train_responses = torch.cat(train_responses, dim=0)
        validation_images = torch.cat(validation_images, dim=0)
        validation_responses = torch.cat(validation_responses, dim=0)
        test_images = torch.cat(test_images, dim=0)
        test_responses = torch.cat(test_responses, dim=0)

        if normalize_across_sessions:
            raise NotImplementedError(
                "Normalization across sessions not implemented yet"
            )

        print("Creating concatenated dataloaders...")
        DefaultDataPoint = namedtuple("DefaultDataPoint", ["images", "responses"])
        image_shape = train_images.shape[-3:]
        concat_train_loader = DataLoader(
            ImageNamedTupleWrappedDataset(
                DefaultDataPoint,
                TensorDataset(train_images, train_responses),
                image_shape=image_shape,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        concat_validation_loader = DataLoader(
            ImageNamedTupleWrappedDataset(
                DefaultDataPoint,
                TensorDataset(validation_images, validation_responses),
                image_shape=image_shape,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        concat_test_loader = DataLoader(
            ImageNamedTupleWrappedDataset(
                DefaultDataPoint,
                TensorDataset(test_images, test_responses),
                image_shape=image_shape,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        # using the session ids of each session, create a concatenated session id for the combined sessions
        combined_session_id = "--".join(list(trainloader.keys()))
        concat_dataloaders = OrderedDict(
            [
                ("train", OrderedDict([(combined_session_id, concat_train_loader)])),
                (
                    "validation",
                    OrderedDict([(combined_session_id, concat_validation_loader)]),
                ),
                ("test", OrderedDict([(combined_session_id, concat_test_loader)])),
            ]
        )
        final_dataloaders = concat_dataloaders
    return final_dataloaders
