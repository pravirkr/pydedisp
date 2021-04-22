#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG_LOAD(value) __ldg(&(value))
#else
#define LDG_LOAD(value) (value)
#endif

// Private helper functions
// ------------------------
template <typename T>
T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
T div_up(T a, T b) {
    return (a - 1) / b + 1;
    // return (a % b != 0) ? (a / b + 1) : (a / b);
}


// Internal abstraction for errors
#if defined(DEDISP_DEBUG) && DEDISP_DEBUG
#define throw_error(error)                                             \
    do {                                                               \
        printf("An error occurred within dedisp on line %d of %s: %s", \
               __LINE__, __FILE__, dedisp_get_error_string(error));    \
        return (error);                                                \
    } while (0)
#define throw_getter_error(error, retval)                              \
    do {                                                               \
        printf("An error occurred within dedisp on line %d of %s: %s", \
               __LINE__, __FILE__, dedisp_get_error_string(error));    \
        return (retval);                                               \
    } while (0)
#else
#define throw_error(error) return error
#define throw_getter_error(error, retval) return retval
#endif  // DEDISP_DEBUG
// ------------------------
