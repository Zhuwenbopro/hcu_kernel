// Each interval ends at the inclusive M in its first column.
constexpr Route routes5120[] = {
    {   4096, launch<5120, Layout::Split8, 256, 3>, launch<5120, Layout::Pair4, 256, 3>},
    {INT_MAX, launch<5120, Layout::Split8, 320, 1>, launch<5120, Layout::Pair4, 256, 0>},
};

constexpr Route routes8704[] = {
    {    256, launch<8704, Layout::Split8, 768, 1>, launch<8704, Layout::Pair4, 512, 1>},
    {   1448, launch<8704, Layout::Split8, 768, 0>, launch<8704, Layout::Pair4, 512, 3>},
    {   8192, launch<8704, Layout::Split8, 512, 3>, launch<8704, Layout::Pair4, 512, 1>},
    {INT_MAX, launch<8704, Layout::Split8, 512, 0>, launch<8704, Layout::Pair4, 256, 1>},
};

constexpr Route routes17408[] = {
    {      2, launch<17408, Layout::Split8, 512, 1>, launch<17408, Layout::Pair4, 768, 0>},
    {     11, launch<17408, Layout::Split8, 512, 2>, launch<17408, Layout::Pair4, 512, 1>},
    {     22, launch<17408, Layout::Split8, 512, 1>, launch<17408, Layout::Pair4, 512, 1>},
    {     39, launch<17408, Layout::Split8, 512, 2>, launch<17408, Layout::Pair4, 512, 1>},
    {     55, launch<17408, Layout::Split8, 512, 2>, launch<17408, Layout::Pair4, 512, 2>},
    {     71, launch<17408, Layout::Split8, 512, 1>, launch<17408, Layout::Pair4, 768, 2>},
    {    110, launch<17408, Layout::Split8, 512, 2>, launch<17408, Layout::Pair4, 768, 2>},
    {    143, launch<17408, Layout::Split8, 512, 2>, launch<17408, Layout::Pair4, 512, 2>},
    {    175, launch<17408, Layout::Split8, 256, 2>, launch<17408, Layout::Pair4, 512, 2>},
    {    214, launch<17408, Layout::Split8, 256, 2>, launch<17408, Layout::Pair4, 256, 2>},
    {    277, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 768, 0>},
    {    357, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 512, 2>},
    {    438, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 512, 3>},
    {    495, launch<17408, Layout::Split8, 256, 0>, launch<17408, Layout::Pair4, 512, 0>},
    {    572, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 512, 2>},
    {    701, launch<17408, Layout::Split8, 256, 0>, launch<17408, Layout::Pair4, 256, 2>},
    {    858, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 512, 2>},
    {    991, launch<17408, Layout::Split8, 256, 0>, launch<17408, Layout::Pair8, 256, 0>},
    {   1144, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 256, 1>},
    {   1402, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 512, 1>},
    {INT_MAX, launch<17408, Layout::Split8, 512, 3>, launch<17408, Layout::Pair4, 256, 1>},
};
