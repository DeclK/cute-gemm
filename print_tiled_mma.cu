#include <cute/tensor.hpp>

using namespace cute;
int main() {
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, 
                        make_layout(Shape<_1, _1, _1>{}), 
                        make_layout(Shape<_1, _1, _1>{})));
    MMA mma_tmp;
    print(mma_tmp);
}