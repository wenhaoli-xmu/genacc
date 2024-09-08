import subprocess
import multiprocessing

def execute_command(gpu_id, layer_idx):
    # Define the command to execute
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train/train_layer_wise.py " \
              f"--env_conf train/genacc19-7.json " \
              f"--layer {layer_idx} " \
              f"--max_oth 1024"
    
    # Execute the command using subprocess
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the process to complete

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--gpus", type=str, default="[0]")
    args = parser.parse_args()

    import math, json
    args.gpus = json.loads(args.gpus)
    num_cycle = int(math.ceil(args.num_layers / len(args.gpus)))

    for cycle in range(0, num_cycle):
        processes = []

        for j in range(0, len(args.gpus)):
            layer_idx = cycle * len(args.gpus) + j

            if layer_idx >= args.num_layers:
                continue

            # Start a new process for each GPU and layer index
            process = multiprocessing.Process(target=execute_command, args=(args.gpus[j], layer_idx))
            process.start()
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process.join()

if __name__ == "__main__":
    main()