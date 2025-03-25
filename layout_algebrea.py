def get_coord(x, shape):
    """ This is a python function to see that how cutlass layouts are working """
    # M is the number of elements in the matrix
    # shape is the shape of the matrix
    coord = []
    for idx, s in enumerate(shape):
        if idx == 0:
            coord_ = x % s
        else:
            prod_s = 1
            for i in range(idx):
                prod_s *= shape[i]
            coord_ = (x // prod_s) % s

        coord.append(coord_)
    return coord


def layout_function(x, shape, stride):
    """ This is a python function to see that how cutlass layouts are working """
    # M is the number of elements in the matrix
    # shape is the shape of the matrix
    # stride is the stride of the matrix
    coord = get_coord(x, shape)
    new_x = 0
    for coord, s in zip(coord, stride):
        new_x += coord * s
    return new_x

def total_elements(shape):
    total_elements = 1
    for s in shape:
        total_elements *= s
    return total_elements

def print_layout_function(shape, stride):
    for i in range(total_elements(shape)):
        print(layout_function(i, shape, stride), end=" ")
    print("")


def complement(M, shape, stride):
    """ calculate the complement of layout """
    c_shape = []
    c_stride = []
    length = len(shape)
    for i in range(length + 1):
        if i == 0:
            c_shape.append(stride[0])
            c_stride.append(1)
        elif i < length:
            c_shape.append(stride[i] // (shape[i - 1] * stride[i - 1]))
            c_stride.append(shape[i - 1] * stride[i - 1])
        elif i == length:
            c_shape.append(M // (shape[i - 1] * stride[i - 1]))
            c_stride.append(shape[i - 1] * stride[i - 1])
    return c_shape, c_stride


print_layout_function([2, 3], [1, 2])
print_layout_function([2, 3], [1, 3])
print_layout_function([2, 3], [1, 4])
c_shape, c_stride = complement(24, [2, 3], [1, 4])
print(c_shape, c_stride)
print_layout_function(c_shape, c_stride)
print_layout_function([2, 2, 3], [1, 2, 4])