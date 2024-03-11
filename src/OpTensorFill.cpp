// SPDX-License-Identifier: Apache-2.0

#include "kompute/operations/OpTensorFill.hpp"
#include "kompute/Tensor.hpp"

namespace kp {

OpTensorFill::OpTensorFill(const std::vector<std::shared_ptr<Tensor>>& tensors, uint32_t data)
{
    KP_LOG_DEBUG("Kompute OpTensorFill constructor with params");

    this->mTensors = tensors;
    this->mData = data;

    for (const auto& tensor : tensors) {
        const uint32_t size = tensor->memorySize();

        if (size % 4 != 0) {
            throw std::runtime_error(fmt::format(
                "Attempting to fill tensors of a size (in bytes) which is not multiple of 4: {}",
                size
            ));
        }
    }
}

OpTensorFill::~OpTensorFill()
{
    KP_LOG_DEBUG("Kompute OpTensorFill destructor started");
}

void
OpTensorFill::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorFill record called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        this->mTensors[i]->recordFill(commandBuffer, this->mData);
    }
}

void
OpTensorFill::preEval(const vk::CommandBuffer& /* commandBuffer */)
{
    KP_LOG_DEBUG("Kompute OpTensorFill preEval called");
}

void
OpTensorFill::postEval(const vk::CommandBuffer& /* commandBuffer */)
{
    KP_LOG_DEBUG("Kompute OpTensorFill postEval called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == kp::Tensor::TensorTypes::eStorage) {
            continue;
        }

        auto& tensor = *this->mTensors[i];
        uint32_t* const rawData = tensor.data<uint32_t>();

        for (size_t i = 0; i < tensor.size(); ++i) {
            rawData[i] = this->mData;
        }
    }
}

} // End of namespace kp

