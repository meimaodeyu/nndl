import paddle
ndim_4_Tensor=paddle.ones([2,3,4,5])
print('Number of dimensions:',ndim_4_Tensor.ndim)
print('Shape of Tensor:',ndim_4_Tensor.shape)
print('Elements number along axis 0 of Tensor:',ndim_4_Tensor.shape[0])
print('Elements number along the last axis of Tensor:',ndim_4_Tensor.shape[-1])
print('Number of elements in Tensor:',ndim_4_Tensor.size)
