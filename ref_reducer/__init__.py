class REFReducer:
    def __init__(self, M):
        self.M = M

    def _swap(self, i, j):
        """Swap rows i and j"""
        self.M[[i, j]] = self.M[[j, i]]

    def _mul(self, i, scalar):
        """Multiply the values in row i by `scalar`"""
        self.M[i] *= scalar

    def _add(self, i, j, scalar):
        """Add the values in row i multipled by `scalar` to row j"""
        row = self.M[i][:] * scalar
        self.M[j] += row

    def reduce(self):
        """Transform the given matrix M to reduced row echelon form.

        The algorithm is roughly as follows:

        1. Set i = 1, j = 1.
        2. Check the ith column of the matrix.
        3. If the ith column is all 0s, set i = 2 and go to step 1.
        4. Otherwise, check if there's a 1 in the jth row of column i. 
        5. If not, check the other entries in column i.
        6. If any of them are equal to 1, swap their row with the jth row of column i.
        7. Otherwise, if none of them equal 1 and there's a 0 in the jth row of column i,
            swap an arbitrary row without a zero in column i with the jth row.
        8. Now there will be a non-zero value in the jth row of column i.
        9. Multiply this row by the reciprocal of that entry. It now has a leading 1.
        10. Clear all other non-zero values (greater than and less than i). Do this by:
        11. For each non-zero entry ki (where k != j), add row j * -M[k][i] to M[k]. 
            This will clear the row.
        12. Now the column is clear. Increment i (moving to the next column) and j
            (indicating that we are now trying to put a leading zero in the next row).
        13. If i > len(M[0]) or j > len(M), we are done.
        14. Otherwise, go to step 2.
        """
        i, j = 0, 0
        M = self.M
        m, n = M.shape

        while i < m and j < n:
            # Check the jth column of M
            all_zeroes = not any(M[i:,j])
            if all_zeroes:
                j += 1
                continue

            k = i
            while k < m:
                if M[k, i] != 0:
                    if k != i:
                        self._swap(k, i)
                    break
                k += 1
            
            self._mul(i, 1 / M[i, j])

            assert M[i, j] == 1

            k = 0
            while k < m:
                if k == i:
                    k += 1
                    continue

                if self.M[k, j] != 0:
                    # Add 
                    self._add(i, k, M[k, j] * -1)
                    assert M[k, j] == 0

                k += 1

            j += 1
            i += 1

        return M


if __name__ == '__main__':
    import numpy as np

    test_M = np.array([[2, 4, 2], [3, 6, 3]], dtype=np.float64)

    reducer = REFReducer(test_M)
    res = reducer.reduce()
    print(res)

    test_m = np.array([
        [1, 2, 3, 8],
        [1, 3, 3, 10],
        [1, 2, 4, 9]
    ], dtype=np.float64)

    reducer = REFReducer(test_m)
    res = reducer.reduce()
    print(res)

