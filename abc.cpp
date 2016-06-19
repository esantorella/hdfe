template <typename T>
pybind11::array_t<T> make_array(const std::vector<size_t>& shape) {
    return pybind11::array(pybind11::buffer_info(
        nullptr,
        sizeof(T),
        pybind11::format_descriptor<T>::value(),
        shape.size(),
        shape,
        calc_strides(shape, sizeof(T))
    ));
}
