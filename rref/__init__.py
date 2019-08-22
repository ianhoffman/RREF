import numpy as np
from collections import namedtuple


Swap = namedtuple('Swap', ['i', 'j'])
Mul = namedtuple('Mul', ['i', 'scalar'])
Add = namedtuple('Add', ['i', 'j', 'scalar'])


def swap(M, i, j):
    M[[i, j]] = M[[j, i]]
    return M


def mul(M, i, scalar):
    M[i] *= scalar
    return M


def add(M, i, j, scalar):
    row = M[j][:]
    M[i] += (row * scalar)
    return M


def rref(M):
    i, j = 0, 0
    M = np.array(M, dtype=np.float64)
    m, n = M.shape
    steps = []

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
                M = swap(M, i, max_idx)
                steps.append(Swap(i, max_idx))

            # Multiply the new row i by the reciprocal of M[i,j]
            # to create a leading 1.
            reciprocal = 1 / M[i, j]
            M = mul(M, i, reciprocal)
            assert M[i, j] == 1
            steps.append(Mul(i, reciprocal))

            # Create zeroes in all other entries in column j
            # by adding to rows k (k = 0...m, k != i) row i 
            # multipled by M[k,j] * -1. Because row i contains
            # a leading 1, this will result in M[k,j] being
            # set equal to 0.
            for k in range(m):
                if k != i and M[k, j] != 0:
                    scalar = M[k, j] * -1
                    M = add(M, k, i, scalar) 
                    assert M[k, j] == 0
                    steps.append(Add(k, i, scalar))

            # Advance to the next column and row.
            i, j = i + 1, j + 1

    # Now we're almost done. As a final touch, shift all
    # all-zero rows to the bottom of the matrix.
    i, j = 0, m - 1
    while i < j:
        if M[i].max() == 0:
            M = swap(M, i, j)
            steps.append(Swap(i, j))
            j -= 1
        else:
            i += 1

    return steps


def print_steps(M, steps):
    M = np.array(M, dtype=np.float64)
    print(f'Putting {M} in reduced row echelon form')
    for op in steps:
        print(f'{op} => ')
        if isinstance(op, Swap):
            i, j = op
            M = swap(M, i, j)
        elif isinstance(op, Mul):
            i, scalar = op
            M = mul(M, i, scalar)
        elif isinstance(op, Add):
            i, j, scalar = op
            M = add(M, i, j, scalar)
        print(M, end='\n\n')


if __name__ == '__main__':
    test_M = [[2, 4, 2], [3, 6, 3]]

    print_steps(test_M, rref(test_M))

    test_M = [
        [1, 2, 3, 8],
        [1, 3, 3, 10],
        [1, 2, 4, 9]
    ]

    print_steps(test_M, rref(test_M))

