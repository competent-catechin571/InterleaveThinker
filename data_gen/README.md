## InterleaveThinker: Raw Data and Interleaved Sequence Construction

### Raw Data Generation

We provide three scripts for raw data generation (**Qwen-Image-Edit-2511**, **FLUX.2-klein-9B**, and **Nano Banana Pro**). 

> **Note:** To accelerate the generation process for Qwen and Klein, you can utilize the API server. Please refer to the **API Service** section for details.

```bash
bash data_gen_klein.sh
bash data_gen_qwen.sh
bash data_gen_nano.sh
```

### Interleaved Sequence Construction

Although our data is primarily designed to train multi-agent pipelines, it can be directly transformed into formal interleaved sequences by injecting the planning step into the generation loop.

We provide two generation modes:
1. **Easy Mode**: Without reflection (contains no failed samples).
2. **Hard Mode**: With reflection.

```bash
python interleave_easy.py
python interleave_reflection.py
```

**Format Specifications:**
- **Easy Mode**: `<gthink></gthink><plan></plan><image></image><info></info> ...loop`
- **Hard Mode**: `<gthink></gthink><plan></plan><image></image> <critic><think></think>...</critic>...loop <info></info> ...loop`