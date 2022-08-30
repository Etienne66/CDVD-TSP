extern "C"{
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define ROUND_OFF 50000

__global__ void kernel_Correlation_rearrange(
    const int width,
    const int height,
    const int channels,
    const int padding,
    const float* input,
    float* output)
  { //blob_rearrange_kernel2
    const int pbottomwidth = width + 2*padding;
    const int pbottomheight = height + 2*padding;
    const int pwidthheight = pbottomwidth * pbottomheight;
    const int widthheight = width * height;
    int xy = blockIdx.x * blockDim.x + threadIdx.x;

    if (xy >= widthheight) {
      return;
    }

    int ch = blockIdx.y;
    int n = blockIdx.z;

    float value = input[(n * channels + ch) * widthheight + xy];

    __syncthreads();

    int xpad = (xy % width + padding);
    int ypad = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    output[(n*pwidthheight+xypad)*channels + ch] = value;
  }



__global__ void kernel_Correlation_updateOutput(
    const int width,
    const int height,
    const int channels,
    const int padding,
    const int kernel_size,
    const int max_displacement,
    const int stride1,
    const int stride2,
    const float* rbot0,
    const float* rbot1,
    float* top)
  { //CorrelateData
    extern __shared__ char patch_data_char[];
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    const int kernel_radius = (kernel_size - 1) / 2;
    const int grid_radius = max_displacement / stride2;
    const int grid_width = grid_radius * 2 + 1;
    const int border_size = max_displacement + kernel_radius;
    const int pbottomwidth = width + 2*padding;
    const int pbottomheight = height + 2*padding;
    const int topChannels = grid_width * grid_width;
    const int topwidth = (pbottomwidth - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomwidth - border_size * 2) / stride1
    const int topheight = (pbottomheight - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomheight - border_size * 2) / stride1
    const int bottomchannels = channels;
    const int bottomwidth = pbottomwidth;
    const int bottomheight = pbottomheight;
    const int topcount = topChannels*topwidth*topheight;

    float *patch_data = (float *)patch_data_char;
    
    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
    int x1 = blockIdx.x*stride1 + max_displacement;
    int y1 = blockIdx.y*stride1 + max_displacement;
    int item = blockIdx.z;
    int ch_off = threadIdx.x;
    
    // Load 3D patch into shared shared memory
    for (int j = 0; j < kernel_size; j++) { // HEIGHT
      for (int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for (int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) {
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = rbot0[idx1];
        }
      }
    }
    
    __syncthreads();
    
    __shared__ float sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
    
    // Compute correlation
    for (int top_channel = 0; top_channel < topChannels; top_channel++) {
      sum[ch_off] = 0;
    
      int s2o = (top_channel % grid_width - grid_radius) * stride2;
      int s2p = (top_channel / grid_width - grid_radius) * stride2;
      
      for (int j = 0; j < kernel_size; j++) { // HEIGHT
        for (int i = 0; i < kernel_size; i++) { // WIDTH
          int ji_off = ((j * kernel_size) + i) * bottomchannels;
          for (int ch = ch_off; ch < bottomchannels; ch += WARPS_PER_BLOCK*THREADS_PER_WARP) {
            int x2 = x1 + s2o;
            int y2 = y1 + s2p;
            
            int idxPatchData = ji_off + ch;
            int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;
            
            sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
          }
        }
      }
      
      __syncthreads();
      
      if (ch_off == 0) {
        float total_sum = 0;
        for (int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
          total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
      }
    }
  }



__global__ void kernel_Correlation_updateGradFirst(
    const int width,
    const int height,
    const int channels,
    const int padding,
    const int kernel_size,
    const int max_displacement,
    const int stride1,
    const int stride2,
    const int pixels, 
    const int intSample,
    const float* rbot1,
    const float* gradOutput,
    float* gradFirst)
  { //CorrelateDataBackward0
    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    const int kernel_radius = (kernel_size - 1) / 2;
    const int grid_radius = max_displacement / stride2;
    const int grid_width = grid_radius * 2 + 1;
    const int border_size = max_displacement + kernel_radius;
    const int pbottomwidth = width + 2*padding;
    const int pbottomheight = height + 2*padding;
    const int topChannels = grid_width * grid_width;
    const int topwidth = (pbottomwidth - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomwidth - border_size * 2) / stride1
    const int topheight = (pbottomheight - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomheight - border_size * 2) / stride1
    const int bottomchannels = channels;
    const int bottomwidth = width;
    const int bottomheight = height;
    const int bottomcount = bottomchannels*bottomwidth*bottomheight;

    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < pixels; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % bottomchannels; // channels
      int l = (intIndex / bottomchannels) % bottomwidth + padding; // w-pos
      int m = (intIndex / bottomchannels / bottomwidth) % bottomheight + padding; // h-pos
      
      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
      int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil(l - 2*kernel_radius - max_displacement) / stride1
      int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil(m - 2*kernel_radius - max_displacement) / stride1
      
      // Same here:
      int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
      int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
      
      float sum = 0;
      if (xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1)) {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);
        
        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);
        
        for (int p = -grid_radius; p <= grid_radius; p++) {
          for (int o = -grid_radius; o <= grid_radius; o++) {
            // Get rbot1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((intSample * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * channels + n;
            float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
            
            // Index offset for gradOutput in following loops:
            int op = (p+grid_radius) * grid_width + (o+grid_radius); // index[o,p]
            int idxopoffset = (intSample * topChannels + op);
            
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgradOutput = (idxopoffset * topheight + y) * topwidth + x; // gradOutput[x,y,o,p]
                sum += gradOutput[idxgradOutput] * bot1tmp;
              }
            }
          }
        }
      }
      const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
      const int bot0index = ((n * bottomheight) + (m-padding)) * bottomwidth  + (l-padding);
      gradFirst[bot0index + intSample*bottomcount] = sum / (float)sumelems;
    }
  }



__global__ void kernel_Correlation_updateGradSecond(
    const int width,
    const int height,
    const int channels,
    const int padding,
    const int kernel_size,
    const int max_displacement,
    const int stride1,
    const int stride2,
    const int pixels,
    const int intSample,
    const float* rbot0,
    const float* gradOutput,
    float* gradSecond)
  { //CorrelateDataBackward1
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    const int kernel_radius = (kernel_size - 1) / 2;
    const int grid_radius = max_displacement / stride2;
    const int grid_width = grid_radius * 2 + 1;
    const int topChannels = grid_width * grid_width;
    const int border_size = max_displacement + kernel_radius;
    const int pbottomwidth = width + 2*padding;
    const int pbottomheight = height + 2*padding;
    const int topwidth = (pbottomwidth - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomwidth - border_size * 2) / stride1
    const int topheight = (pbottomheight - border_size * 2 + round_off_s1 - 1) / stride1 + 1 - round_off;// ceil(pbottomheight - border_size * 2) / stride1
    const int bottomchannels = channels;
    const int bottomwidth = width;
    const int bottomheight = height;
    const int bottomcount = bottomchannels*bottomwidth*bottomheight;

    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < pixels; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % bottomchannels; // channels
      int l = (intIndex / bottomchannels) % bottomwidth + padding; // w-pos
      int m = (intIndex / bottomchannels / bottomwidth) % bottomheight + padding; // h-pos
      
      float sum = 0;
      for (int p = -grid_radius; p <= grid_radius; p++) {
        for (int o = -grid_radius; o <= grid_radius; o++) {
          int s2o = stride2 * o;
          int s2p = stride2 * p;
          
          //Get X,Y ranges and clamp
          // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
          int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
          int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (m - 2*kernel_radius - max_displacement - s2p) / stride1
          
          // Same here:
          int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
          int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1
            
          if (xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1)) {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);
            
            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);
            
            // Get rbot0 data:
            int idxbot0 = ((intSample * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            float bot0tmp = rbot0[idxbot0]; // rbot0[l+s2o,m+s2p,n]
            
            // Index offset for gradOutput in following loops:
            int op = (p+grid_radius) * grid_width + (o+grid_radius); // index[o,p]
            int idxopoffset = (intSample * topChannels + op);
            
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgradOutput = (idxopoffset * topheight + y) * topwidth + x; // gradOutput[x,y,o,p]
                sum += gradOutput[idxgradOutput] * bot0tmp;
              }
            }
          }
        }
      }
      const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
      const int bot1index = ((n * bottomheight) + (m-padding)) * bottomwidth  + (l-padding);
      gradSecond[bot1index + intSample*bottomcount] = sum / (float)sumelems;
    } 
  }
}
