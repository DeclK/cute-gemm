#include <cute/tensor.hpp>
using namespace cute;

int main() {
    using T = cute::half_t;
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                            make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                        make_stride(Int<4>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{}))));
    G2SCopyA g2s_copy_a;
    print(g2s_copy_a);
}