
def tuple_to_linear(tup, M, N, L=4, index0=False):
    '''Convert a tuple format index of a qubit in an (N, M, L) processor
    to linear format'''

    qpr = 2*N*L     # qbits per row
    qpt = 2*L       # qbits per tile

    return (0 if index0 else 1) + qpr*tup[0]+qpt*tup[1]+L*tup[2]+tup[3]


def linear_to_tuple(ind, M, N, L=4, index0=False):
    '''Convert the linear index of a qubit in an (N, M, L) processor to
    tuple format'''

    qpr = 2*N*L     # qbits per row
    qpt = 2*L       # qbits per tile

    if not index0:
        ind -= 1

    row, rem = divmod(ind, qpr)
    col, rem = divmod(rem, qpt)
    horiz, ind = divmod(rem, L)

    return (row, col, horiz, ind)
