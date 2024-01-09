#include "execution_event.h"
#include "execution_queue.h"

namespace SPH::execution
{
ExecutionEvent::ExecutionEvent(const sycl::event &event) : event_list_{event} {}
ExecutionEvent::ExecutionEvent(const std::vector<sycl::event> &event) : event_list_{event} {}
ExecutionEvent &ExecutionEvent::operator=(sycl::event event)
{
    event_list_.emplace_back(std::move(event));
    return *this;
}
void ExecutionEvent::wait()
{
    sycl::event::wait(event_list_);
    event_list_.clear();
}
const std::vector<sycl::event> &ExecutionEvent::getEventList() const
{
    return event_list_;
}
ExecutionEvent &ExecutionEvent::add(sycl::event event)
{
    event_list_.emplace_back(std::move(event));
    return *this;
}
ExecutionEvent &ExecutionEvent::add(const SPH::execution::ExecutionEvent &event)
{
    auto new_events = event.getEventList();
    event_list_.insert(event_list_.end(), new_events.begin(), new_events.end());
    return *this;
}
ExecutionEvent &ExecutionEvent::then(std::function<void()> &&func,
                                     std::optional<std::reference_wrapper<ExecutionEvent>> host_event)
{
    auto host_sycl_event =
        executionQueue.getQueue()
            .submit([&](sycl::handler &cgh)
                    {
                        cgh.depends_on(getEventList());
                        cgh.host_task(std::move(func)); });
    if(host_event)
        host_event.value().get() = std::move(host_sycl_event);
    return *this;
}

} // namespace SPH::execution
