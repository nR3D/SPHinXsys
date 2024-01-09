#include "execution_queue.h"

SPH::execution::ExecutionQueue &SPH::execution::ExecutionQueue::getInstance()
{
    static ExecutionQueue instance;
    return instance;
}
sycl::queue &SPH::execution::ExecutionQueue::getQueue()
{
    if(!sycl_queue)
        sycl_queue = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
    return *sycl_queue;
}
size_t SPH::execution::ExecutionQueue::getWorkGroupSize() const
{
    return work_group_size;
}
void SPH::execution::ExecutionQueue::setWorkGroupSize(size_t workGroupSize)
{
    work_group_size = workGroupSize;
}
sycl::nd_range<1> SPH::execution::ExecutionQueue::getUniformNdRange(size_t global_size, size_t local_size)
{
    return {global_size % local_size ? (global_size / local_size + 1) * local_size : global_size , local_size};
}
sycl::nd_range<1> SPH::execution::ExecutionQueue::getUniformNdRange(size_t global_size) const
{
    // sycl::nd_range is trivially-copyable, no std::move required
    return getUniformNdRange(global_size, work_group_size);
}
