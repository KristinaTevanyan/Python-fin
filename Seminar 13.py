
def rectangle_part():
    rect_a = RectangleWorks.rectangle_create(100, 100)      # stub
    rect_b = RectangleWorks.rectangle_create(100, 100)      # stub

    try:
        rect_a = RectangleWorks.rectangle_create(11, -14)
    except RectangleValueError as exc:
        print(f'{exc.__class__.__name__}: {exc}')
    try:
        rect_b = RectangleWorks.rectangle_create(14, 'a')
    except RectangleValueError as exc:
        print(f'{exc.__class__.__name__}: {exc}')
    try:
        rect_e = RectangleWorks.rectangle_sum(rect_a, 10)
    except RectangleTypeError as exc:
        print(f'{exc.__class__.__name__}: {exc}')
    try:
        rect_e = RectangleWorks.rectangle_sub(rect_a, 'a')
    except RectangleTypeError as exc:
        print(f'{exc.__class__.__name__}: {exc}')
    rect_c = RectangleWorks.rectangle_sum(rect_a, rect_b)
    rect_d = RectangleWorks.rectangle_sub(rect_a, rect_b)
    print(rect_c)
    print(rect_d)


def matrix_part():
    mtx_a = MatrixWorks.create_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    mtx_b = MatrixWorks.create_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    mtx_d = MatrixWorks.create_matrix([[1, 2, 3, 4, ], [5, 6, 7, 8], [9, 10, 11, 12]])
    try:
        mtx_c = MatrixWorks.create_matrix([[10, 11, 12], [4, 5, 6, 100], [1, 2, 3], [7, 8, 9]])
    except ConsistencyMatrixError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(MatrixWorks.matrices_sum(mtx_a, mtx_d))
    except MatrixValueError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(MatrixWorks.matrices_mul(mtx_a, mtx_b))
    except MatrixMultiplyError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(MatrixWorks.matrices_mul(mtx_a, mtx_d))
    except MatrixMultiplyError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(MatrixWorks.matrices_mul(mtx_a, 10))
    except MatrixValueError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(MatrixWorks.matrices_mul(mtx_a, 'a'))
    except MatrixTypeError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')


def file_lister_part():
    start_path = '/home/andrew/Documents/geekbrains/Python2023/Homeworks/Seminar_13'
    wrong_path = 'Documents/geekbrains/Python2023/Homeworks/Seminar_13'
    try:
        dir_walker = FileListerWorks.create_lister(wrong_path)
    except FileListerPathError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
        dir_walker = FileListerWorks.create_lister(start_path)
    try:
        print(FileListerWorks.list_dir('json', False))
    except FileListerObjectError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(FileListerWorks.list_dir(dir_walker, False))
    except FileListerExtensionError as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(FileListerWorks.list_dir(False))
    except (FileListerObjectError, FileListerExtensionError) as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')
    try:
        print(FileListerWorks.list_dir(dir_walker, 'py'))
    except (FileListerObjectError, FileListerExtensionError) as exc:
        print(f'FAIL! {exc.__class__.__name__}: {exc}')


def main():
    rectangle_part()
    matrix_part()
    file_lister_part()


if __name__ == '__main__':
    main()