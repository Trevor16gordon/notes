
# Block And Grid Size


Example:

Image is 255 by 255


if block_size is (24, 24, 1) and grid size is (1, 1, 1)

blockIdx.x: Stays at 0
blockIdx.y: Stays at 0
blockDim.x: Stays at 24
blockDim.y: Stays at 24
threadIdx.x: Goes from 0 to 23
threadIdx.y: Goes from 0 to 23


if block_size is (24, 24, 1) and grid size is (21, 21, 1)

blockIdx.x: Goes from 0 to 20
blockIdx.y: Goes from 0 to 20
blockDim.x: Stays at 24
blockDim.y: Stays at 24
threadIdx.x: Goes from 0 to 23
threadIdx.y: Goes from 0 to 23
