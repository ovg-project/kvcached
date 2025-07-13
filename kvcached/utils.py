import logging

PAGE_SIZE = 2 * 1024 * 1024  # 2MB


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells


def get_kvcached_logger(name: str = 'kvcached') -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if none exists (prevents duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[kvcached][%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to inference engines; avoid duplicate messages
        logger.propagate = False

    return logger
