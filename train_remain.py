import subprocess
import multiprocessing

def execute_command(gpu_id, layer_idx):
    # Define the command to execute
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train/train_layer_wise.py " \
              f"--env_conf train/genacc19-6.json " \
              f"--layer {layer_idx} " \
              f"--max_oth 1024"
    
    # Execute the command using subprocess
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the process to complete

def main():
    for gpu_id, layer_idx in enumerate(range(26, 32, 8), start=0):
        processes = []

        # Start a new process for each GPU and layer index
        process = multiprocessing.Process(target=execute_command, args=(gpu_id, layer_idx))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()