#define __HIP_PLATFORM_AMD__

#include <hip/hip_fp16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



namespace thrust {
namespace detail {

// Make __half* known to Thrust's pointer traits
template<>
struct pointer_raw_pointer<__half*>
{
    typedef __half* type;
};

template<>
struct pointer_raw_pointer<const __half*>
{
    typedef const __half* type;
};

// (Optional, but harmless) Also for float, though usually already supported
template<>
struct pointer_raw_pointer<float*>
{
    typedef float* type;
};

template<>
struct pointer_raw_pointer<const float*>
{
    typedef const float* type;
};

} // namespace detail
} 