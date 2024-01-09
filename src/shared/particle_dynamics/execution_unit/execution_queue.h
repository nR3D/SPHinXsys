#ifndef SPHINXSYS_EXECUTION_QUEUE_H
#define SPHINXSYS_EXECUTION_QUEUE_H

#include <sycl/sycl.hpp>

namespace SPH::execution {
    class ExecutionQueue {
    public:
        ExecutionQueue(ExecutionQueue const&) = delete;
        void operator=(ExecutionQueue const&) = delete;

        static ExecutionQueue& getInstance();

        sycl::queue &getQueue();

        size_t getWorkGroupSize() const;

        void setWorkGroupSize(size_t workGroupSize);

        static sycl::nd_range<1> getUniformNdRange(size_t global_size, size_t local_size);

        sycl::nd_range<1> getUniformNdRange(size_t global_size) const;

    private:
        ExecutionQueue() : work_group_size(32), sycl_queue() {}

        std::size_t work_group_size;
        std::unique_ptr<sycl::queue> sycl_queue;

    } static &executionQueue = ExecutionQueue::getInstance();
}

#endif //SPHINXSYS_EXECUTION_QUEUE_H
