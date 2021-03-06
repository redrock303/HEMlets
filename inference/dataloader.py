import math

import torch




def val_loader(dataset, config, local_rank, num_gpu):
    num_data = len(dataset)
    data_per_gpu = math.ceil(num_data / num_gpu)
    st = local_rank * data_per_gpu
    ed = min(num_data, st + data_per_gpu)
    indices = range(st, ed)
    subset = torch.utils.data.Subset(dataset, indices)

    sampler = torch.utils.data.sampler.SequentialSampler(subset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, config.VAL.IMG_PER_GPU, drop_last=False)

    loader = torch.utils.data.DataLoader(subset, num_workers=config.VAL.NUM_WORKERS, batch_sampler=batch_sampler)

    return loader


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from exps.baseline.config import config


    from dataset import get_train_dataset, get_val_dataset
    dataset = get_train_dataset(config)
    # dataset = get_val_dataset(config)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # loader = train_loader(dataset, config)
    loader = val_loader(dataset, config, 0, 2)

    iter_loader = iter(loader)
    if args.local_rank == 0:
        lr, hr = iter_loader.next()
        print(lr.size(), hr.size())
