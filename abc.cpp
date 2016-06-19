inline std::vector<size_t> calc_strides(
    const std::vector<size_t>& shape, size_t unit_size)
{
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = unit_size;
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
