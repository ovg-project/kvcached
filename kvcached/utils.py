PAGE_SIZE = 2 * 1024 * 1024  # 2MB


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells
