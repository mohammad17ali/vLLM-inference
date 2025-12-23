### Building the Docker Image
 docker build -f Dockerfile -t vllm-openvino-env .

### Run in Interactive Terminal (-it)
docker run -it --rm vllm-openvino-env

#### Inside the container:
> upgrade apt (no sudo needed)
> root@abcxyz:/workspace# sudo apt-get update
> install python3
> root@abcxyz:/workspace# apt-get install -y python3-pip python3-venv git wget
> update the alternatives, such that reference to python points to python3
> root@abcxyz:/workspace# update-alternatives --install /usr/bin/python python /usr/bin/python3 1
> set OpenVino config
> root@b30248a7245f:/workspace# export LLM_OPENVINO_DEVICE=CPU
> root@b30248a7245f:/workspace# export VLLM_OPENVINO_KVCACHE_SPACE=8
> root@b30248a7245f:/workspace# export VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8
> root@b30248a7245f:/workspace# export VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON

#### Run sample inference
root@b30248a7245f:/workspace# 
`python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
sp = SamplingParams(max_tokens=32, temperature=0.7, top_p=0.9)
outs = llm.generate(["Hello from OpenVINO + vLLM"], sp)
print(outs[0].outputs[0].text)
PY`