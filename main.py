import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn import linear_model
import time

from lsm_weight_definitions import initWeights1
from modules.lsm_models import LSM

if __name__ == "__main__":
    # Load dataset (Using NMNIST here)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=3000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )

    # Load NMNIST train and test datasets
    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )

    # Cache datasets to disk for faster loading
    cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
    cached_testset = DiskCachedDataset(testset, cache_path="./cache/nmnist/test")

    # Create data loaders with padding collation
    batch_size = 256
    trainloader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=True,
    )
    testloader = DataLoader(
        cached_testset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

    # Set device to MPS (Metal Performance Shaders) if available, otherwise CPU
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Get sample batch to determine input size
    data, targets = next(iter(trainloader))
    flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))
    print("data shape: ", data.shape)
    print("flat data shape: ", flat_data.shape)

    in_sz = flat_data.shape[-1]

    # Set neuron parameters for LSM
    tauV = 16.0  # Voltage time constant
    tauI = 16.0  # Current time constant
    th = 20  # Spike threshold
    curr_prefac = np.float32(1 / tauI)  # Current prefactor
    alpha = np.float32(np.exp(-1 / tauI))  # Exponential decay for current
    beta = np.float32(1 - 1 / tauV)  # Exponential decay for voltage

    # Initialize LSM weights
    Nz = 10  # Number of neurons per layer
    Win, Wlsm = initWeights1(27, 2, 0.15, in_sz, Nz=Nz)
    abs_W_lsm = np.abs(Wlsm)
    print("average fan out: ", np.mean(np.sum(abs_W_lsm > 0, axis=1)))

    # Create and move LSM network to device
    N = Wlsm.shape[0]
    lsm_net = LSM(
        N,
        in_sz,
        np.float32(curr_prefac * Win),
        np.float32(curr_prefac * Wlsm),
        alpha=alpha,
        beta=beta,
        th=th,
    ).to(device)

    num_partitions = 3
    lsm_net.eval()

    # Process training data through LSM and collect spike outputs
    with torch.no_grad():
        start_time = time.time()
        for i, (data, targets) in enumerate(iter(trainloader)):
            if i % 25 == 24:
                print("train batches completed: ", i)
            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(
                device
            )
            part_steps = flat_data.shape[0] // num_partitions
            spk_rec = lsm_net(flat_data)

            # Partition spike output and compute mean across partitions
            if i == 0:
                lsm_parts = []
                for part in range(num_partitions):
                    lsm_parts.append(
                        torch.mean(
                            spk_rec[part * part_steps : (part + 1) * part_steps], dim=0
                        )
                    )
                lsm_out = torch.cat(lsm_parts, dim=1)
                in_train = torch.mean(flat_data, dim=0).cpu().numpy()
                lsm_out_train = lsm_out.cpu().numpy()
                lsm_label_train = np.int32(targets.numpy())
            else:
                lsm_parts = []
                for part in range(num_partitions):
                    lsm_parts.append(
                        torch.mean(
                            spk_rec[part * part_steps : (part + 1) * part_steps], dim=0
                        )
                    )
                lsm_out = torch.cat(lsm_parts, dim=1)
                in_train = np.concatenate(
                    (in_train, torch.mean(flat_data, dim=0).cpu().numpy()), axis=0
                )
                lsm_out_train = np.concatenate(
                    (lsm_out_train, lsm_out.cpu().numpy()), axis=0
                )
                lsm_label_train = np.concatenate(
                    (lsm_label_train, np.int32(targets.numpy())), axis=0
                )
        end_time = time.time()

        print("running time of training epoch: ", end_time - start_time, "seconds")

        # Process test data through LSM and collect spike outputs
        for i, (data, targets) in enumerate(iter(testloader)):
            if i % 25 == 24:
                print("test batches completed: ", i)
            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(
                device
            )
            part_steps = flat_data.shape[0] // num_partitions
            spk_rec = lsm_net(flat_data)

            # Partition spike output and compute mean across partitions
            if i == 0:
                lsm_parts = []
                for part in range(num_partitions):
                    lsm_parts.append(
                        torch.mean(
                            spk_rec[part * part_steps : (part + 1) * part_steps], dim=0
                        )
                    )
                lsm_out = torch.cat(lsm_parts, dim=1)
                in_test = torch.mean(flat_data, dim=0).cpu().numpy()
                lsm_out_test = lsm_out.cpu().numpy()
                lsm_label_test = np.int32(targets.numpy())
            else:
                lsm_parts = []
                for part in range(num_partitions):
                    lsm_parts.append(
                        torch.mean(
                            spk_rec[part * part_steps : (part + 1) * part_steps], dim=0
                        )
                    )
                lsm_out = torch.cat(lsm_parts, dim=1)
                in_test = np.concatenate(
                    (in_test, torch.mean(flat_data, dim=0).cpu().numpy()), axis=0
                )
                lsm_out_test = np.concatenate(
                    (lsm_out_test, lsm_out.cpu().numpy()), axis=0
                )
                lsm_label_test = np.concatenate(
                    (lsm_label_test, np.int32(targets.numpy())), axis=0
                )

    # Print output shapes and statistics
    print(lsm_out_train.shape)
    print(lsm_out_test.shape)

    print(in_train.shape)
    print(in_test.shape)

    print("mean in spiking (train) : ", np.mean(in_train))
    print("mean in spiking (test) : ", np.mean(in_test))

    print("mean LSM spiking (train) : ", np.mean(lsm_out_train))
    print("mean LSM spiking (test) : ", np.mean(lsm_out_test))

    # Train linear classifier on LSM outputs
    print("training linear model:")
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(lsm_out_train, lsm_label_train)

    # Evaluate classifier on test set
    score = clf.score(lsm_out_test, lsm_label_test)
    print("test score = " + str(score))
