from dataclasses import dataclass

@dataclass
class Layout:
    shape: list
    stride: list

    def size(self):
        total_elements = 1
        for s in self.shape:
            total_elements *= s
        return total_elements

    def get_coord(self, x):
        if x >= self.size():
            print("[Warning] x is out of range")
        coord = []
        for idx, s in enumerate(self.shape):
            if idx == 0:
                coord_ = x % s
            else:
                prod_s = 1
                for i in range(idx):
                    prod_s *= self.shape[i]
                coord_ = (x // prod_s) % s
            coord.append(coord_)
        return coord

    def layout_function(self, x):
        coord = self.get_coord(x)
        new_x = 0
        for c, s in zip(coord, self.stride):
            new_x += c * s
        return new_x

def print_layout_function(layout: Layout):
    for i in range(layout.size()):
        print(layout.layout_function(i), end=" ")
    print("")

def complement(M, layout: Layout):
    shape, stride = layout.shape, layout.stride
    length = len(shape)
    for i in range(1, length):
        if stride[i] % (shape[i-1] * stride[i-1]) != 0:
            raise ValueError("Layout not admissible: stride[i] not divisible by shape[i-1] * stride[i-1]")
    if M % (shape[-1] * stride[-1]) != 0:
        raise ValueError("Layout not admissible: M not divisible by shape[-1] * stride[-1]")
    c_shape = []
    c_stride = []
    for i in range(length + 1):
        if i == 0:
            c_shape.append(stride[0])
            c_stride.append(1)
        elif i < length:
            c_shape.append(stride[i] // (shape[i-1] * stride[i-1]))
            c_stride.append(shape[i-1] * stride[i-1])
        elif i == length:
            c_shape.append(M // (shape[i-1] * stride[i-1]))
            c_stride.append(shape[i-1] * stride[i-1])
    return Layout(c_shape, c_stride)

def prod(lst):
    p = 1
    for x in lst:
        p *= x
    return p

def find_division_index(S, r):
    prod_i = 1
    for i in range(len(S)):
        if r % prod_i == 0:
            c = r // prod_i
            if 1 <= c < S[i] and S[i] % c == 0:
                return i, c
        prod_i *= S[i]
    if r % prod_i == 0:
        return len(S), r // prod_i
    raise ValueError("M is not left divisible by r")

def compose_layouts_length1(A, B):
    if len(B.shape) != 1 or len(B.stride) != 1:
        raise ValueError("B must be a length 1 layout")
    N, r = B.shape[0], B.stride[0]
    S, D = A.shape, A.stride
    i, c = find_division_index(S, r)
    if i < len(S):
        M_prime = S[i] // c
        if N <= M_prime:
            return Layout([N], [c * D[i]])
        else:
            prod = M_prime
            shape = [M_prime]
            strides = [c * D[i]]
            j = i + 1
            while j < len(S):
                if N % prod == 0:
                    c_prime = N // prod
                    if c_prime < S[j] or j == len(S) - 1:
                        if c_prime > 1:
                            shape.append(c_prime)
                            strides.append(D[j])
                        return Layout(shape, strides)
                    elif c_prime == 1:
                        return Layout(shape, strides)
                prod *= S[j]
                shape.append(S[j])
                strides.append(D[j])
                j += 1
            if N % prod == 0:
                c_prime = N // prod
                if c_prime > 1:
                    shape.append(c_prime)
                    strides.append(D[len(S)-1])
                return Layout(shape, strides)
            raise ValueError("N does not satisfy weak left divisibility")
    else:
        return Layout([N], [c * D[-1]])

def compose_layouts(A, B):
    S, D = A.shape, A.stride
    if len(B.shape) == 1:
        return compose_layouts_length1(A, B)
    modes = [Layout([B.shape[k]], [B.stride[k]]) for k in range(len(B.shape))]
    for mode in modes:
        try:
            compose_layouts_length1(A, mode)
        except ValueError:
            raise ValueError("A and B_k not admissible for composition")
    M_prime = prod(S[:-1]) if len(S) > 1 else 1
    J_list = []
    for mode in modes:
        r_k, N_k = mode.stride[0], mode.shape[0]
        if N_k > 1 and r_k <= M_prime:
            start = max(r_k, 1)
            end = min(r_k * (N_k - 1), M_prime)
            if start <= end:
                J_list.append((start, end))
            else:
                J_list.append(None)
        else:
            J_list.append(None)
    for i in range(len(J_list)):
        for j in range(i + 1, len(J_list)):
            if J_list[i] and J_list[j]:
                a, b = J_list[i]
                c, d = J_list[j]
                if not (b < c or d < a):
                    raise ValueError("Intervals of definition overlap")
    composed = [compose_layouts_length1(A, mode) for mode in modes]
    shape = []
    stride = []
    for cm in composed:
        shape.extend(cm.shape)
        stride.extend(cm.stride)
    return Layout(shape, stride)

def logical_division(A, B):
    M = A.size()
    try:
        c_B = complement(M, B)
        concatenated = Layout(B.shape + c_B.shape, B.stride + c_B.stride)
        return compose_layouts(A, concatenated)
    except ValueError as e:
        raise ValueError(f"Logical division failed: {e}")

# Test your examples
print_layout_function(Layout([2, 3], [1, 2]))  # 0 1 2 3 4 5
print_layout_function(Layout([2, 3], [1, 3]))  # 0 1 3 4 6 7
print_layout_function(Layout([2, 3], [1, 4]))  # 0 1 4 5 8 9
c_layout = complement(24, Layout([2, 3], [1, 4]))
print_layout_function(c_layout)  # 0 1 2 3
print_layout_function(Layout([2, 2, 3], [1, 2, 4]))  # 0 1 2 3 4 5 6 7 8 9 10 11

layout_A = Layout([6,2], [8,2])
layout_B = Layout([4,3], [3,1])
composed = compose_layouts(layout_A, layout_B)
print(composed)

layout_A = Layout([20], [2])
layout_B = Layout([5,4], [4,1])
composed = compose_layouts(layout_A, layout_B)
print(composed)

layout_A = Layout([10,2], [16,4])
layout_B = Layout([5,4], [1,5])
composed = compose_layouts(layout_A, layout_B)
print(composed)