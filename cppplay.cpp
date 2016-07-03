<%
setup_pybind11(cfg)
%>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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

template <typename T>
pybind11::array_t<T> make_array(const std::vector<size_t>& shape) {
    return pybind11::array(pybind11::buffer_info(
        nullptr,
        sizeof(T),
        pybind11::format_descriptor<T>::value,
        shape.size(),
        shape,
        calc_strides(shape, sizeof(T))
    ));
}



struct groupby {
    std::vector<std::vector<int>> indices;

    groupby(pybind11::array_t<int> keys) {
        auto keys_buf = keys.request();
        int* keys_begin = reinterpret_cast<int*>(keys_buf.ptr);
        int* keys_end = keys_begin + keys_buf.shape[0];
        auto result_it = std::max_element(keys_begin, keys_end);
        int n_keys = *result_it + 1;
        indices.resize(n_keys);
        for (size_t i = 0; i < keys_buf.shape[0]; i++) {
            indices[keys_begin[i]].push_back(i);	
        }
    }
    
    pybind11::array_t<double> apply_2(pybind11::array_t<double> y,
                                      pybind11::object f) {
        auto y_buf = y.request();
        pybind11::array_t<double> out = make_array<double>({y_buf.shape[0]});
        auto out_buf = out.request();
        for (size_t i = 0; i < indices.size(); i++) {
//            pybind11::array_t<double> group_input = 
//                make_array<double>({indices[i].size()});
//            auto group_input_buf = group_input.request();
//            for (size_t j = 0; j < indices[i].size(); j++) {
//                reinterpret_cast<double*>(group_input_buf.ptr)[j] = 
//                    reinterpret_cast<double*>(y_buf.ptr)[indices[i][j]];
//            }
//            double group_output = f(group_input).cast<double>();
            double mean = 0.0;
            for (size_t j = 0; j < indices[i].size(); j++) {
                mean += reinterpret_cast<double*>(y_buf.ptr)[indices[i][j]];
            }
            mean /= indices[i].size();
            double group_output = mean;
            for (size_t j = 0; j < indices[i].size(); j++) {
                reinterpret_cast<double*>(out_buf.ptr)[indices[i][j]] =
                    group_output;
            }
        }
        return out;
    }

    void print_indices() {
        for (size_t i = 0; i < indices.size(); i++) {
            std::cout << "index " << i;
            for (int elt: indices[i]) {
                std::cout << " " << elt;
            }
            std::cout << std::endl;
        }
        
    }
};

PYBIND11_PLUGIN(cppplay) {
    pybind11::module m("cppplay", "auto-compiled c++ extension");
    pybind11::class_<groupby>(m, "Groupby")
        .def(pybind11::init<pybind11::array_t<int>>())
        .def("apply_2", &groupby::apply_2);
    return m.ptr();
}
