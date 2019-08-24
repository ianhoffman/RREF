import numpy as np


def rref(M):
    i, j = 0, 0
    M = np.array(M, dtype=np.float64)
    m, n = M.shape

    def print_matrix():
        print(np.around(M), end='\n\n')

    def swap(i, j):
        print(f'Swap({i}, {j}) => ')
        M[[i, j]] = M[[j, i]]
        print_matrix()

    def mul(i, scalar):
        print(f'Mul(row={i}, scalar={scalar}) => ')
        M[i] *= scalar
        print_matrix()

    def add(i, j, scalar):
        print(f'Add(row={i}, row_to_add={j}, scalar={scalar}) => ')
        row = M[j][:]
        M[i] += (row * scalar)
        print_matrix()

    print('=' * 70)
    print('')
    print('Beginning Gauss-Jordan Elimination => ')
    print_matrix()

    while i < m and j < n:
        # Go to the next column if this column has all zeroes.
        # We only need to consider from the ith row down because
        # the rows above row i already have leading 1s and so
        # should not be swapped around.
        if M[i:,j].max() == 0:
            j += 1
        else:
            # Get the maximum (obviously nonzero) value in this
            # column and swap its row with row i.
            max_idx = M[i:,j].argmax() 
            if max_idx > i:
                swap(i, max_idx)

            # Multiply the new row i by the reciprocal of M[i,j]
            # to create a leading 1.
            reciprocal = 1 / M[i, j]
            mul(i, reciprocal)

            # Create zeroes in all other entries in column j
            # by adding to rows k (k = 0...m, k != i) row i 
            # multipled by M[k,j] * -1. Because row i contains
            # a leading 1, this will result in M[k,j] being
            # set equal to 0.
            for k in range(m):
                if k != i and M[k, j] != 0:
                    scalar = M[k, j] * -1
                    add(k, i, scalar) 

            # Advance to the next column and row.
            i, j = i + 1, j + 1

    # Now we're almost done. As a final touch, shift all
    # all-zero rows to the bottom of the matrix.
    i, j = 0, m - 1
    while i < j:
        if M[i].max() == 0:
            M = swap(i, j)
            j -= 1
        else:
            i += 1

    return M


if __name__ == '__main__':
    test_M = [[2, 4, 2], [3, 6, 3]]

    rref(test_M)

    test_M = [
        [1, 2, 3, 8],
        [1, 3, 3, 10],
        [1, 2, 4, 9]
    ]

    rref(test_M)

    test_M = [
        [1, 1, 0, 3],
        [2, 3, 4, 2]
    ]

    rref(test_M)

    test_M = [
        [1, 1, 1],
        [2, -1, 5],
        [3, 4, 2]
    ]

    rref(test_M)

    test_M = [
        [4, 3, 2, -1, 4],
        [5, 4, 3, -1, 4],
        [-2, -2, -1, 2, -3],
        [11, 6, 4, 1, 11]
    ]

    rref(test_M)

    test_M = [
        [3, 6, 9, 5, 25, 53],
        [7, 14, 21, 9, 53, 105],
        [-4, -8, -12, 5, 10, 11]
    ]

    rref(test_M)
