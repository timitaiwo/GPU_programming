# https://www.youtube.com/live/BBhZ9Ltpmdw?si=svPKBwIzFXZ0vYRK

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext

def print_threads():
    print("Block index:", block_idx.x, "\t", "Thread index: ", thread_idx.x)

def main():
    try:
        # create device context
        ctx = DeviceContext()

        # GPU kernel launches asynchronously - doesn't block host CPU
        ctx.enqueue_function[print_threads](grid_dim=2, block_dim=4)

        # synchronize Host thread with GPU
        ctx.synchronize()

    except:
        print("failed")
