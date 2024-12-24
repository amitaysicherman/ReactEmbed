# sbatch --gres=gpu:L40:8 --mem=256G -c16 --time=0-01:00 --wrap="python preprocessing/multy_gpu_seq_to_vec.py --model GearNet"
import os
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as mp

from preprocessing.seq_to_vec import SeqToVec, model_to_type, fill_none_with_zeros


def setup_device(gpu_id):
    """Set up the device for the current process"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    return device


def process_batch(batch_data, model_name, gpu_id, data_name):
    """Process a batch of sequences on a specific GPU"""
    device = setup_device(gpu_id)

    # Initialize model for this process
    seq_to_vec = SeqToVec(model_name)
    seq_to_vec.model = seq_to_vec.model.to(device)

    batch_vectors = []
    for seq in batch_data:
        if len(seq.strip()) == 0:
            batch_vectors.append(None)
        else:
            vec = seq_to_vec.to_vec(seq.strip())
            batch_vectors.append(vec)

    return batch_vectors


def parallel_main(model, data_name, batch_size=32):
    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA devices available")

    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)

    # Load data
    data_types = model_to_type(model)
    file = f'data/{data_name}/{"proteins" if data_types == "protein" else "molecules"}_sequences.txt'

    with open(file, "r") as f:
        lines = f.readlines()

    # Split data into batches
    n_samples = len(lines)
    n_batches = (n_samples + batch_size - 1) // batch_size
    batches = [lines[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]

    # Create output file path
    output_file = f"data/{data_name}/{model}_vectors.npy"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
        return None

    # Process batches in parallel
    all_vectors = []
    with mp.Pool(n_gpus) as pool:
        process_func = partial(process_batch, model_name=model, data_name=data_name)

        # Distribute batches across GPUs
        for batch_idx, batch in enumerate(batches):
            gpu_id = batch_idx % n_gpus
            process_func_with_gpu = partial(process_func, gpu_id=gpu_id)

            # Process batch
            batch_vectors = pool.apply_async(process_func_with_gpu, (batch,))
            all_vectors.extend(batch_vectors.get())

    # Fill None values with zeros
    all_vectors = fill_none_with_zeros(all_vectors)
    all_vectors = np.array(all_vectors)

    # Save results
    np.save(output_file, all_vectors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Parallel sequence to vector conversion')
    parser.add_argument('--model', type=str, help='Model to use', default="MolCLR",
                        choices=["ProtBert", "ChemBERTa", "MoLFormer", "esm3-small",
                                 "esm3-medium", "GearNet", "MolCLR"])
    parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU', default=150)

    args = parser.parse_args()
    if "esm3" in args.model:
        pass
    if "GearNet" in args.model:
        pass
    if args.model == "MolCLR":
        pass

    parallel_main(args.model, args.data_name, args.batch_size)
