from . import device_env  # noqa: F401 — before any torch CUDA init

from .run import main

if __name__ == "__main__":
    main()
