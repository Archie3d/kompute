#pragma once

#include "kompute/Core.hpp"
#include "kompute/operations/OpBase.hpp"

namespace kp {

class OpTensorFill : public OpBase
{
public:

    OpTensorFill(const std::vector<std::shared_ptr<Tensor>>& tensors, uint32_t data = 0);
    ~OpTensorFill() override;
    void record(const vk::CommandBuffer& commandBuffer) override;
    void preEval(const vk::CommandBuffer& commandBuffer) override;
    void postEval(const vk::CommandBuffer& commandBuffer) override;

private:
    std::vector<std::shared_ptr<Tensor>> mTensors{};
    uint32_t mData{};
};

} // End namespace kp
