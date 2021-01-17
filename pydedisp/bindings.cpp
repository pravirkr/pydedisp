#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <dedisp/dedisp.hpp>
#include <dedisp/DedispPlan.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

// Based on https://github.com/pybind/pybind11/issues/2271
namespace array_utils {
template <typename T>
inline pybind11::array_t<T> get_ndarray(const T* in_array,
                                        dedisp_size arr_size) {
    auto out
        = py::array_t<T>({arr_size}, {sizeof(T)}, in_array, py::cast(in_array));
    return out;
}

// Based on https://github.com/pybind/pybind11/issues/2121
template <typename T>
inline pybind11::array_t<T> get_ndarray_readonly(const T* in_array,
                                                 dedisp_size arr_size) {
    auto out
        = py::array_t<T>({arr_size}, {sizeof(T)}, in_array, py::cast(in_array));
    out.attr("flags").attr("writeable") = false;
    return out;
}
}  // namespace array_utils

PYBIND11_MODULE(_libdedisp, mod) {
    mod.doc()               = "dedisp class functions";
    //mod.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);

    py::class_<DedispPlan> clsDedispPlan(mod, "DedispPlan");
    // Copy constructor
    //clsDedispPlan.def(py::init<const DedispPlan&>(), "other"_a);
    // Default constructor
    clsDedispPlan.def(
        py::init<dedisp_size, dedisp_float, dedisp_float, dedisp_float>(),
        "nchans"_a, "dt"_a, "f0"_a, "df"_a);

    clsDedispPlan
        .def_property_readonly("gulp_size",
                               [](const DedispPlan& plan) -> dedisp_size {
                                   return plan.get_gulp_size();
                               })
        .def_property_readonly("max_delay",
                               [](const DedispPlan& plan) -> dedisp_size {
                                   return plan.get_max_delay();
                               })
        .def_property_readonly("channel_count",
                               [](const DedispPlan& plan) -> dedisp_size {
                                   return plan.get_channel_count();
                               })
        .def_property_readonly("dm_count",
                               [](const DedispPlan& plan) -> dedisp_size {
                                   return plan.get_dm_count();
                               })
        .def_property_readonly("dt",
                               [](const DedispPlan& plan) -> dedisp_float {
                                   return plan.get_dt();
                               })
        .def_property_readonly("df",
                               [](const DedispPlan& plan) -> dedisp_float {
                                   return plan.get_df();
                               })
        .def_property_readonly("f0",
                               [](const DedispPlan& plan) -> dedisp_float {
                                   return plan.get_f0();
                               })

        .def_property_readonly(
            "dm_list",
            [](const DedispPlan& plan) -> py::array_t<dedisp_float> {
                return array_utils::get_ndarray_readonly<dedisp_float>(
                    plan.get_dm_list(), plan.get_dm_count());
            },
            py::return_value_policy::move)

        .def_property_readonly(
            "killmask",
            [](const DedispPlan& plan) -> py::array_t<dedisp_bool> {
                return array_utils::get_ndarray_readonly<dedisp_bool>(
                    plan.get_killmask(), plan.get_channel_count());
            },
            py::return_value_policy::move)

        .def_static("set_device", &DedispPlan::set_device)

        .def("set_gulp_size",
             [](DedispPlan& plan, dedisp_size gulp_size) {
                 plan.set_gulp_size(gulp_size);
             })

        .def("set_dm_list",
             [](DedispPlan& plan, const py::array_t<dedisp_float>& dm_list) {
                 dedisp_size count = dm_list.shape(0);
                 plan.set_dm_list(
                     reinterpret_cast<const dedisp_float*>(dm_list.data()),
                     count);
             })

        .def("set_killmask",
             [](DedispPlan& plan, const py::array_t<dedisp_bool>& killmask) {
                 plan.set_killmask(
                     reinterpret_cast<const dedisp_bool*>(killmask.data()));
             })

        .def("generate_dm_list",
             [](DedispPlan& plan, dedisp_float dm_start, dedisp_float dm_end,
                dedisp_float ti, dedisp_float tol) {
                 plan.generate_dm_list(dm_start, dm_end, ti, tol);
             })

        .def("execute",
             [](DedispPlan& plan, dedisp_size nsamps,
                const py::array_t<dedisp_byte>& in, dedisp_size in_nbits,
                py::array_t<dedisp_byte>& out, dedisp_size out_nbits,
                unsigned flags) {
                 plan.execute(
                     nsamps, reinterpret_cast<const dedisp_byte*>(in.data()),
                     in_nbits,
                     reinterpret_cast<dedisp_byte*>(out.mutable_data()),
                     out_nbits, flags);
             })

        .def("execute_adv",
             [](DedispPlan& plan, dedisp_size nsamps,
                const py::array_t<dedisp_byte>& in, dedisp_size in_nbits,
                dedisp_size in_stride, py::array_t<dedisp_byte>& out,
                dedisp_size out_nbits, dedisp_size out_stride, unsigned flags) {
                 plan.execute_adv(
                     nsamps, reinterpret_cast<const dedisp_byte*>(in.data()),
                     in_nbits, in_stride,
                     reinterpret_cast<dedisp_byte*>(out.mutable_data()),
                     out_nbits, out_stride, flags);
             })

        .def("execute_guru",
             [](DedispPlan& plan, dedisp_size nsamps,
                const py::array_t<dedisp_byte>& in, dedisp_size in_nbits,
                dedisp_size in_stride, py::array_t<dedisp_byte>& out,
                dedisp_size out_nbits, dedisp_size out_stride,
                dedisp_size first_dm_idx, dedisp_size dm_count,
                unsigned flags) {
                 plan.execute_guru(
                     nsamps, reinterpret_cast<const dedisp_byte*>(in.data()),
                     in_nbits, in_stride,
                     reinterpret_cast<dedisp_byte*>(out.mutable_data()),
                     out_nbits, out_stride, first_dm_idx, dm_count, flags);
             });
}
